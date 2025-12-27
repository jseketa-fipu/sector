#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-}"
DOMAIN="${DOMAIN:-sector.seketa.it}"
EMAIL="${EMAIL:-you@seketa.it}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-/etc/rancher/k3s/k3s.yaml}"
REPO_DIR="${REPO_DIR:-/opt/sector-sim}"
K3S_VERSION="${K3S_VERSION:-}"

if [ -z "$REPO_URL" ]; then
  echo "Usage: $0 <git_repo_url>" >&2
  exit 1
fi

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "Please run as root (sudo) so k3s can install." >&2
  exit 1
fi

install_packages() {
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

if ! command -v git >/dev/null 2>&1; then
  install_packages git
fi
if ! command -v curl >/dev/null 2>&1; then
  install_packages curl
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "Installing k3s (includes kubectl)..."
  if [ -n "$K3S_VERSION" ]; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -
  else
    curl -sfL https://get.k3s.io | sh -
  fi
fi

export KUBECONFIG="$KUBECONFIG_PATH"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found after k3s install." >&2
  exit 1
fi

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

if [ ! -d "$REPO_DIR/.git" ]; then
  mkdir -p "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --rebase
fi

# Install NGINX ingress (bundled with the repo).
kubectl apply -f "$REPO_DIR/k8s/ingress-nginx/controller.yaml"

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml

# Wait for cert-manager deployment
kubectl rollout status deploy/cert-manager -n cert-manager --timeout=120s
kubectl rollout status deploy/cert-manager-webhook -n cert-manager --timeout=120s
kubectl rollout status deploy/cert-manager-cainjector -n cert-manager --timeout=120s

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

# Apply app manifests
kubectl apply -k "$REPO_DIR/k8s"

# Patch ingresses for host + TLS.
kubectl -n sector-sim patch ingress sector-sim-viz --type=merge -p "{\"spec\":{\"tls\":[{\"hosts\":[\"${DOMAIN}\"],\"secretName\":\"sector-seketa-it-tls\"}],\"rules\":[{\"host\":\"${DOMAIN}\"}]}}"
kubectl -n sector-sim patch ingress sector-sim-api --type=merge -p "{\"spec\":{\"tls\":[{\"hosts\":[\"${DOMAIN}\"],\"secretName\":\"sector-seketa-it-tls\"}],\"rules\":[{\"host\":\"${DOMAIN}\"}]}}"

cat <<EOF

Deployment kicked off.
- DNS A record: ${DOMAIN} -> this VPS public IP
- Check ingress: kubectl get ingress -n sector-sim
- Check cert: kubectl get certificate -n sector-sim

EOF
