# Deploying Cambrian

This guide covers three deployment targets for the Cambrian evolution server and API agents:

- **Render** — simplest managed deployment, zero infrastructure management
- **Docker** — portable container for any cloud or on-prem host
- **Kubernetes** — production-grade orchestration for teams running at scale

---

## Prerequisites

All deployment targets assume you have:

- An OpenAI-compatible API key (`OPENAI_API_KEY`)
- An evolved genome exported with `cambrian export` or `export_genome_json()`
- Python 3.11+ (or Docker installed)

---

## 1. Render

[Render](https://render.com) is the fastest path to a live Cambrian API with zero infrastructure setup.

### Step 1: Export your agent

```bash
# Evolve and save the best genome
cambrian evolve "Summarise scientific papers" \
    --generations 20 \
    --population 10 \
    --output best_genome.json

# Export as a FastAPI application
python -c "
from cambrian.agent import Agent
from cambrian.export import load_genome_json, export_api
genome = load_genome_json('best_genome.json')
from cambrian.agent import Agent
agent = Agent(genome=genome)
export_api(agent, 'api_agent.py')
"
```

### Step 2: Create a `requirements.txt`

```
cambrian>=1.0.2
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
```

### Step 3: Create a `render.yaml`

```yaml
services:
  - type: web
    name: cambrian-agent
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api_agent:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false        # set via Render dashboard, not committed to git
      - key: CAMBRIAN_LOG_LEVEL
        value: INFO
```

### Step 4: Deploy

1. Push your repo to GitHub
2. Connect the repo to Render → **New Web Service**
3. Set `OPENAI_API_KEY` in the **Environment** tab
4. Click **Deploy**

Your agent will be live at `https://cambrian-agent.onrender.com/run`.

### Testing the deployed endpoint

```bash
curl -X POST https://cambrian-agent.onrender.com/run \
     -H "Content-Type: application/json" \
     -d '{"task": "Summarise the theory of evolution in 3 sentences."}'
```

---

## 2. Docker

Docker gives you a portable, reproducible image that runs identically on any host.

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the exported agent and genome
COPY api_agent.py .
COPY best_genome.json .

# Expose API port
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "api_agent:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build and run locally

```bash
# Build
docker build -t cambrian-agent:latest .

# Run with your API key
docker run --rm \
    -e OPENAI_API_KEY="sk-..." \
    -p 8080:8080 \
    cambrian-agent:latest
```

### Push to a registry and deploy

```bash
# Tag and push to Docker Hub (or ECR, GCR, etc.)
docker tag cambrian-agent:latest your-registry/cambrian-agent:1.0.0
docker push your-registry/cambrian-agent:1.0.0

# On any host with Docker installed:
docker run -d \
    -e OPENAI_API_KEY="sk-..." \
    -p 80:8080 \
    --restart unless-stopped \
    your-registry/cambrian-agent:1.0.0
```

### docker-compose.yml (with Redis cache, optional)

```yaml
version: "3.9"

services:
  cambrian-agent:
    image: cambrian-agent:latest
    ports:
      - "8080:8080"
    environment:
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      CAMBRIAN_LOG_LEVEL: "INFO"
    restart: unless-stopped
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    restart: unless-stopped
```

---

## 3. Kubernetes

For production workloads that need auto-scaling, rolling updates, and health checks.

### Deployment manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cambrian-agent
  labels:
    app: cambrian-agent
    version: "1.0.0"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cambrian-agent
  template:
    metadata:
      labels:
        app: cambrian-agent
    spec:
      containers:
        - name: cambrian-agent
          image: your-registry/cambrian-agent:1.0.0
          ports:
            - containerPort: 8080
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: cambrian-secrets
                  key: openai-api-key
            - name: CAMBRIAN_LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 30
```

### Service manifest

```yaml
apiVersion: v1
kind: Service
metadata:
  name: cambrian-agent-svc
spec:
  selector:
    app: cambrian-agent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```

### Secret for the API key

```bash
kubectl create secret generic cambrian-secrets \
    --from-literal=openai-api-key="sk-..."
```

### Ingress (nginx)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cambrian-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
    - host: cambrian.your-domain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: cambrian-agent-svc
                port:
                  number: 80
```

### Deploy

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Watch rollout
kubectl rollout status deployment/cambrian-agent

# Scale horizontally
kubectl scale deployment/cambrian-agent --replicas=5
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cambrian-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cambrian-agent
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## Health endpoint

All export templates include a `/health` endpoint that returns `{"status": "ok"}`.
Configure your load balancer or probe to hit this endpoint.

---

## Environment variables reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for the LLM backend |
| `OPENAI_API_BASE` | No | `https://api.openai.com/v1` | Override for Anthropic, Groq, Ollama, etc. |
| `CAMBRIAN_LOG_LEVEL` | No | `WARNING` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CAMBRIAN_BASE_URL` | No | — | Alias for `OPENAI_API_BASE` |

See [ENV_VARS.md](ENV_VARS.md) for the complete reference including per-provider examples.

---

## Running Cambrian CLI in a container

You can also run the full evolution loop in a container:

```bash
docker run --rm \
    -e OPENAI_API_KEY="sk-..." \
    -v "$(pwd)/output:/app/output" \
    cambrian-agent:latest \
    cambrian evolve "Solve maths problems step by step" \
        --generations 10 \
        --output /app/output/best.json
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Agent has no backend` | `OPENAI_API_KEY` not set | Set the env var; see above |
| 502 Bad Gateway | Container not ready | Check readiness probe logs; increase `initialDelaySeconds` |
| High memory usage | Large population or many few-shot examples | Reduce `--population` or trim `few_shot_examples` in genome |
| Slow cold start on Render | Free tier spins down | Upgrade to paid instance; or add `/health` ping-warmup |
| `OPENAI` error 429 | Rate limit exceeded | Add `--workers 1` to uvicorn; implement request queuing |
