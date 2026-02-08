#!/usr/bin/env bash
# Deploy the stack onto a single VPS using k3s + nginx ingress.
# This script installs prerequisites, installs/validates k3s, builds a local image,
# loads it into k3s' containerd, and applies the k8s manifests.
set -euo pipefail

# --- Configuration (can be overridden via environment variables) ---
REPO_URL="${1:-}"
DOMAIN="${DOMAIN:-sector.seketa.it}"
EMAIL="${EMAIL:-you@seketa.it}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-/opt/sector-sim}"
K3S_VERSION="${K3S_VERSION:-}"
IMAGE_NAME="${IMAGE_NAME:-distributed-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
STOP_HOST_NGINX="${STOP_HOST_NGINX:-1}"
TLS_SECRET_NAME="${TLS_SECRET_NAME:-sector-tls}"

if [ -z "$REPO_URL" ]; then
  # First arg is required so we can clone/pull the repo.
  echo "Usage: $0 <git_repo_url>" >&2
  exit 1
fi

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  # k3s install and system package installs require root.
  echo "Please run as root (sudo) so k3s can install." >&2
  exit 1
fi

install_packages() {
  # OS-agnostic package install helper (apt/yum/apk).
  local packages=("$@")
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y "${packages[@]}"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "${packages[@]}"
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache "${packages[@]}"
  else
    echo "No supported package manager found. Install: ${packages[*]}" >&2
    exit 1
  fi
}

install_docker() {
  # Best-effort Docker installation across distros.
  if command -v apt-get >/dev/null 2>&1; then
    install_packages docker.io
  elif command -v yum >/dev/null 2>&1; then
    install_packages docker
  elif command -v apk >/dev/null 2>&1; then
    install_packages docker
  else
    echo "No supported package manager found. Install Docker manually." >&2
    exit 1
  fi
}

if ! command -v git >/dev/null 2>&1; then
  # Git is needed to clone the repo.
  install_packages git
fi
if ! command -v curl >/dev/null 2>&1; then
  # Curl is used for k3s install and external manifests.
  install_packages curl
fi
if ! command -v docker >/dev/null 2>&1; then
  # Docker builds the app image locally.
  install_docker
fi
if command -v systemctl >/dev/null 2>&1; then
  # Ensure Docker is running so we can build images.
  systemctl enable --now docker >/dev/null 2>&1 || true
fi

if ! command -v kubectl >/dev/null 2>&1; then
  # Install k3s if kubectl is missing (k3s bundles kubectl).
  echo "Installing k3s (includes kubectl)..."
  if [ -n "$K3S_VERSION" ]; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -
  else
    curl -sfL https://get.k3s.io | sh -
  fi
fi

export KUBECONFIG="$KUBECONFIG_PATH"

if ! command -v kubectl >/dev/null 2>&1; then
  # Sanity check after k3s install.
  echo "kubectl not found after k3s install." >&2
  exit 1
fi

# --- Wait for the k3s node to be Ready ---
# Wait for node to be Ready.
for i in {1..30}; do
  if kubectl get nodes >/dev/null 2>&1; then
    if kubectl get nodes | grep -q " Ready "; then
      break
    fi
  fi
  sleep 2
  if [ "$i" -eq 30 ]; then
    echo "k3s node not ready." >&2
    exit 1
  fi
done

# Always clone fresh to ensure latest content is used for the build.
if [ -d "$REPO_DIR" ]; then
  rm -rf "$REPO_DIR"
fi
mkdir -p "$REPO_DIR"
git clone "$REPO_URL" "$REPO_DIR"

# --- Host cleanup to free ports 80/443 for k3s servicelb ---
# Free host ports 80/443 if requested (needed for k3s servicelb).
if [ "$STOP_HOST_NGINX" = "1" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl stop nginx >/dev/null 2>&1 || true
  systemctl disable nginx >/dev/null 2>&1 || true
fi

# --- Remove Traefik to avoid port conflicts with nginx ingress ---
# Remove Traefik so nginx ingress can bind 80/443.
if kubectl -n kube-system get helmchart traefik >/dev/null 2>&1; then
  kubectl -n kube-system delete helmchart traefik || true
fi
kubectl -n kube-system delete svc traefik >/dev/null 2>&1 || true
kubectl -n kube-system delete deploy traefik >/dev/null 2>&1 || true

# --- Build the app image locally and load into k3s containerd ---
# Build locally and load into k3s containerd to avoid external registries.
# Ensure we don't reuse a stale image layer.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker rmi -f "$IMAGE" >/dev/null 2>&1 || true
fi
docker build --no-cache -t "$IMAGE" "$REPO_DIR"
tmp_image_tar="$(mktemp)"
docker save -o "$tmp_image_tar" "$IMAGE"
# Remove any existing k3s image with the same tag to avoid stale layers.
k3s ctr images rm "$IMAGE" >/dev/null 2>&1 || true
k3s ctr images import "$tmp_image_tar"
rm -f "$tmp_image_tar"

# --- Install the nginx ingress controller shipped with the repo ---
# Install NGINX ingress (bundled with the repo).
kubectl apply -f "$REPO_DIR/k8s/ingress-nginx/controller.yaml"

# --- Wait for k3s servicelb to bind 80/443 to the node ---
# Wait for k3s servicelb to bind the ingress controller.
for i in {1..30}; do
  if kubectl -n kube-system get pods \
    -l svccontroller.k3s.cattle.io/svcname=ingress-nginx-controller \
    >/dev/null 2>&1; then
    if kubectl -n kube-system get pods \
      -l svccontroller.k3s.cattle.io/svcname=ingress-nginx-controller \
      -o jsonpath='{.items[*].status.phase}' | grep -q "Running"; then
      break
    fi
  fi
  sleep 2
done

# --- Install cert-manager for TLS (Let’s Encrypt) ---
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

# --- Wait for cert-manager to be ready ---
# Wait for cert-manager deployment
kubectl rollout status deploy/cert-manager -n cert-manager --timeout=120s
kubectl rollout status deploy/cert-manager-webhook -n cert-manager --timeout=120s
kubectl rollout status deploy/cert-manager-cainjector -n cert-manager --timeout=120s

# --- Create a ClusterIssuer for Let's Encrypt ---
# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt
spec:
  acme:
    email: ${EMAIL}
    server: https://acme-v02.api.letsencrypt.org/directory
    privateKeySecretRef:
      name: letsencrypt-account-key
    solvers:
      - http01:
          ingress:
            class: nginx
EOF

# --- Apply app manifests (Redis, API, workers, ingresses) ---
# Apply app manifests
kubectl apply -k "$REPO_DIR/k8s"

# --- Patch ingress host + TLS without wiping HTTP paths ---
# Patch ingress host + TLS without wiping HTTP paths.
kubectl -n sector-sim patch ingress sector-sim-frontend --type=json -p="[
  {\"op\":\"replace\",\"path\":\"/spec/tls/0/hosts/0\",\"value\":\"${DOMAIN}\"},
  {\"op\":\"replace\",\"path\":\"/spec/rules/0/host\",\"value\":\"${DOMAIN}\"},
  {\"op\":\"replace\",\"path\":\"/spec/tls/0/secretName\",\"value\":\"${TLS_SECRET_NAME}\"}
]"
kubectl -n sector-sim patch ingress sector-sim-api --type=json -p="[
  {\"op\":\"replace\",\"path\":\"/spec/tls/0/hosts/0\",\"value\":\"${DOMAIN}\"},
  {\"op\":\"replace\",\"path\":\"/spec/rules/0/host\",\"value\":\"${DOMAIN}\"},
  {\"op\":\"replace\",\"path\":\"/spec/tls/0/secretName\",\"value\":\"${TLS_SECRET_NAME}\"}
]"

# --- Guard: ensure ingress paths still exist (older scripts used merge patches that wiped rules) ---
frontend_paths="$(kubectl -n sector-sim get ingress sector-sim-frontend -o jsonpath='{.spec.rules[0].http.paths[*].path}' || true)"
api_paths="$(kubectl -n sector-sim get ingress sector-sim-api -o jsonpath='{.spec.rules[0].http.paths[*].path}' || true)"
if [ -z "$frontend_paths" ] || [ -z "$api_paths" ]; then
  echo "Ingress paths missing; reapplying ingress manifests..." >&2
  kubectl apply -f "$REPO_DIR/k8s/base/ingress-frontend.yaml"
  kubectl apply -f "$REPO_DIR/k8s/base/ingress-api.yaml"
  kubectl -n sector-sim patch ingress sector-sim-frontend --type=json -p="[
    {\"op\":\"replace\",\"path\":\"/spec/tls/0/hosts/0\",\"value\":\"${DOMAIN}\"},
    {\"op\":\"replace\",\"path\":\"/spec/rules/0/host\",\"value\":\"${DOMAIN}\"},
    {\"op\":\"replace\",\"path\":\"/spec/tls/0/secretName\",\"value\":\"${TLS_SECRET_NAME}\"}
  ]"
  kubectl -n sector-sim patch ingress sector-sim-api --type=json -p="[
    {\"op\":\"replace\",\"path\":\"/spec/tls/0/hosts/0\",\"value\":\"${DOMAIN}\"},
    {\"op\":\"replace\",\"path\":\"/spec/rules/0/host\",\"value\":\"${DOMAIN}\"},
    {\"op\":\"replace\",\"path\":\"/spec/tls/0/secretName\",\"value\":\"${TLS_SECRET_NAME}\"}
  ]"
fi

cat <<EOF

Deployment kicked off.
- DNS A record: ${DOMAIN} -> this VPS public IP
- Check ingress: kubectl get ingress -n sector-sim
- Check cert: kubectl get certificate -n sector-sim

EOF
