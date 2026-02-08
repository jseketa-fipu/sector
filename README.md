# Sector: 40k‑Inspired Simulation

A playable, Warhammer 40k–inspired sector simulation with real-time updates and bot factions. The stack is Redis-backed and fully stateless at the service layer:
- API service (auth, orders, snapshot, events, admin controls)
- Simulation worker (advances the world and emits snapshots/events)
- Frontend worker (serves the UI and streams updates over WebSocket)
- Bot worker (AI orders for unclaimed factions)
- NFT minter (standalone HTTP service for minting awards)

All long-lived state lives in Redis (streams + snapshot hash).

## Running locally

```bash
# build and start redis, api, sim-worker, bot-worker, and frontend
docker compose up --build
```

API will be on `http://localhost:8000` (health at `/health`, snapshot at `/snapshot`). Frontend is on `http://localhost:9000/`. Redis is exposed on `localhost:6379`.

The NFT minter is not part of docker-compose. It has its own Dockerfile for testing:

```bash
docker build -t sector-nft-minter:local services/nft_minter
docker run --rm -p 9100:9100 sector-nft-minter:local
```

## Deploying on Kubernetes

The `k8s/` directory contains a kustomize base for Redis, API, sim worker, frontend worker, bot worker, and ingress. Deployment can be done end-to-end with the script in `deploy_vps.sh`.

Deploy script usage (VPS):
- Prereqs: root/sudo on the VPS, outbound internet access, and a DNS name pointing to the VPS.
- Required: git repo URL as the first argument.
- Required env vars: `DOMAIN`, `EMAIL`.
- Optional env vars: `KUBECONFIG_PATH`, `REPO_DIR`, `K3S_VERSION`, `IMAGE_NAME`, `IMAGE_TAG`, `STOP_HOST_NGINX`, `TLS_SECRET_NAME`.
- Run:

```bash
DOMAIN=sector.example.com EMAIL=you@example.com ./deploy_vps.sh <git_repo_url>
```

Manual deployment (without the script):
1. Build the application image and push it to a registry your cluster can reach:
   ```bash
   docker build -t <registry>/distributed-app:latest .
   docker push <registry>/distributed-app:latest
   ```
2. Update the image reference via kustomize (either edit `k8s/kustomization.yaml` or run `kustomize edit set image distributed-app=<registry>/distributed-app:latest`).
3. Install the bundled NGINX ingress controller (creates the `ingress-nginx` namespace, RBAC, Deployment, and a `LoadBalancer` Service):
   ```bash
   kubectl apply -f k8s/ingress-nginx/controller.yaml
   # wait for the ingress controller pod to be Ready and note the EXTERNAL-IP on its Service
   kubectl get pods,svc -n ingress-nginx
   ```
   (If your cluster lacks a load balancer, edit the Service type in `k8s/ingress-nginx/controller.yaml` to `NodePort` and expose the ports manually.)
4. Apply the app manifests (includes separate ingress objects for API and frontend; frontend HTTP traffic is served at `/`, and `/ws` is served by the same ingress):
   ```bash
   kubectl apply -k k8s
   ```
5. Add `sector.local` to your hosts file (pointing to `127.0.0.1` on Docker Desktop) and hit `http://sector.local/api/health` or `http://sector.local/`. Without an ingress controller you can still port-forward the ClusterIP services:
   ```bash
   kubectl port-forward svc/api 8000:8000
   kubectl port-forward svc/frontend 9000:9000
   ```
6. Tear everything down with:
   ```bash
   kubectl delete -k k8s
   kubectl delete -f k8s/ingress-nginx/controller.yaml
   ```

## Configuration

- Shared Redis:
  - `REDIS_URL` points services at Redis.
  - `EVENT_STREAM`, `ORDER_STREAM`, `SNAPSHOT_KEY` name the Redis keys/streams.
- API:
  - `PORT`, `JWT_SECRET`, `JWT_TTL_SECONDS`, `AUTH_NONCE_TTL_SECONDS`
  - `BOT_API_TOKEN` (required by bot-worker to post orders)
- Sim worker:
  - `LEASE_TTL_MS`, `ORDER_BLOCK_MS`, `RESET_UNIVERSE_ON_START`
  - `EVENT_STREAM_MAXLEN`, `SNAPSHOT_EVERY`
  - `WAIT_FOR_FRONTEND`, `FRONTEND_HEALTH_URL`
- Bot worker:
  - `BOT_API_URL`, `BOT_API_TOKEN`, `BOT_POLL_INTERVAL`, `BOT_MAX_ORDERS`, `BOT_EVENT_BLOCK_MS`
- Viz:
  - `API_URL`, `PORT`
- NFT minter:
  - `PORT`, `MINTER_API_KEY`

## How it works (stateless loop)

1. Sim worker acquires a Redis lease to ensure only one tick loop runs.
2. Sim worker loads or seeds the world, then consumes orders from the orders stream.
3. Orders are normalized (pathing, fleet selection), applied, and the world advances a tick.
4. Sim worker publishes snapshot + events to Redis (snapshot hash + event stream).
5. API serves snapshot/events and accepts orders. Human orders require auth; bots use `X-Bot-Token`.
6. Frontend worker serves the UI and a `/ws` stream sourced from Redis events.
7. Bot worker polls snapshots, generates AI orders for unclaimed factions, and posts them to the API.

## API endpoints

Base URL: `http://localhost:8000` (via API service). The frontend worker proxies API calls under `/api/*` for the browser.

Auth:
- `POST /auth/nonce` to get a nonce + login message.
- `POST /auth/verify` with signature to get a JWT.
- Pass `Authorization: Bearer <token>` on human-only endpoints.
- Bot orders use `X-Bot-Token: <BOT_API_TOKEN>` and skip human auth.

Endpoints (API service):
- `GET /health`
- `POST /auth/nonce`
- `POST /auth/verify`
- `GET /me`
- `GET /factions`
- `POST /factions/claim`
- `GET /snapshot`
- `GET /events?after=<id>&count=<n>`
- `POST /orders`
- `POST /admin/restart`
- `POST /admin/bots-only`
- `GET /admin/pause`
- `POST /admin/pause`

NFT minter (separate service, default `http://localhost:9100`):
- `GET /health`
- `POST /mint` (optional `Authorization: Bearer <MINTER_API_KEY>`)

## Architecture diagram (human player timeline)

```mermaid
sequenceDiagram
  autonumber
  actor Player
  participant Frontend as Web Frontend
  participant API as API Service
  participant Redis as Redis
  participant Sim as Sim Worker
  participant Bot as Bot Worker

  Player->>Frontend: Open UI
  Frontend->>API: GET /snapshot
  API->>Redis: Load snapshot
  Redis-->>API: Snapshot
  API-->>Frontend: Snapshot
  Frontend-->>Player: Render world

  Player->>Frontend: Connect live updates
  Frontend->>API: WebSocket /ws (via frontend worker proxy)
  API->>Redis: Stream events
  Redis-->>API: Events
  API-->>Frontend: Events
  Frontend-->>Player: Live updates

  Player->>Frontend: Request login
  Frontend->>API: POST /auth/nonce
  API-->>Frontend: Nonce + message
  Player->>Frontend: Sign message
  Frontend->>API: POST /auth/verify (signature)
  API-->>Frontend: JWT

  Player->>Frontend: Claim faction
  Frontend->>API: POST /factions/claim (Bearer JWT)
  API->>Redis: Record faction claim
  API-->>Frontend: Claim result

  Player->>Frontend: Issue order(s)
  Frontend->>API: POST /orders (Bearer JWT)
  API->>Redis: Append to order stream

  alt Unclaimed factions exist
    Bot->>Redis: Load snapshot
    Bot->>Bot: Generate AI orders
    Bot->>API: POST /orders (X-Bot-Token)
    API->>Redis: Append to order stream
  else All factions claimed by humans
    Bot-->>Bot: No orders this tick
  end

  Sim->>Redis: Read orders
  Sim->>Sim: Apply orders, advance tick
  Sim->>Redis: Save snapshot + append events
  API->>Redis: Read events/snapshot
  API-->>Frontend: Events / snapshot
  Frontend-->>Player: World updates
```

## Architecture diagram

```mermaid
flowchart LR
  subgraph Clients
    FRONTEND[Web Frontend]
    BOT[Bot Service]
  end

  API[API Service]
  WORKER[Sim Worker]
  BOTW[Bot Worker]
  REDIS[(Redis)]
  MINTER[NFT Minter]

  FRONTEND -- snapshot/events/ws --> API
  BOT -- orders --> API
  API -- read/write --> REDIS
  WORKER -- read/lease/orders --> REDIS
  WORKER -- snapshot/events --> REDIS
  BOTW -- snapshot --> REDIS
  BOTW -- orders --> API
  API -- mint requests --> MINTER
```

## Architecture diagram (deployment)

```mermaid
flowchart TB
  subgraph Kubernetes
    INGRESS[Ingress - NGINX]
    API_SVC[Service: api]
    FRONTEND_SVC[Service: frontend]
    API_POD[Deployment: api]
    FRONTEND_POD[Deployment: frontend]
    WORKER_POD[Deployment: sim-worker]
    BOT_POD[Deployment: bot-worker]
    REDIS_POD[StatefulSet: redis]
  end

  INGRESS --> API_SVC --> API_POD
  INGRESS --> FRONTEND_SVC --> FRONTEND_POD
  WORKER_POD --> REDIS_POD
  BOT_POD --> REDIS_POD
  API_POD --> REDIS_POD
```

## Architecture diagram (code structure)

```mermaid
flowchart LR
  subgraph Services
    API_CODE[services/api]
    WORKER_CODE[services/sim_worker]
    FRONTEND_CODE[services/frontend_worker]
    BOT_CODE[services/bot_worker]
    MINTER_CODE[services/nft_minter]
  end

  subgraph Simulation
    WORLD[sector/world.py]
    PUPPET[sector/puppet.py]
    STATE[sector/state_utils.py]
    MODELS[sector/models.py]
    STREAMS[sector/infra/redis_streams.py]
  end

  API_CODE --> STATE
  API_CODE --> STREAMS
  WORKER_CODE --> WORLD
  WORKER_CODE --> PUPPET
  WORKER_CODE --> STATE
  WORKER_CODE --> STREAMS
  FRONTEND_CODE --> STATE
  FRONTEND_CODE --> STREAMS
  BOT_CODE --> PUPPET
  BOT_CODE --> STREAMS
  MINTER_CODE --> MODELS
  WORLD --> MODELS
  PUPPET --> MODELS
  STATE --> MODELS
```

## Next steps

- Add metrics/observability and resilience (backoff, retries).
