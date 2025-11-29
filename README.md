# Deployment Guide

## Local Deployment with Docker

This project is configured for local deployment using Docker and Docker Compose.

### Quick Start
```bash
# Set environment variable
export WANDB_API_KEY="your_wandb_api_key"

# Run both services
docker-compose up --build

# Access the application:
# - Frontend: http://localhost:8501
# - Backend API: http://localhost:8080
# - API Docs: http://localhost:8080/docs
```

---

## Architecture
```
┌─────────────────────────────────────────┐
│        Docker Compose Network           │
│                                          │
│  ┌────────────────────────────────┐    │
│  │  Frontend (Streamlit)          │    │
│  │  Port: 8501                    │    │
│  │  Image: fraud-frontend         │    │
│  └────────┬───────────────────────┘    │
│           │ HTTP Requests               │
│           ▼                              │
│  ┌────────────────────────────────┐    │
│  │  Backend (FastAPI)             │    │
│  │  Port: 8080                    │    │
│  │  Image: fraud-backend          │    │
│  │  Loads: model.pkl, scaler.pkl  │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)
- Git (to clone the repository)
- Weights & Biases API key ([Get key](https://wandb.ai/authorize))

---

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Set Environment Variables
```bash
# Export W&B API key
export WANDB_API_KEY="your_wandb_api_key_here"

# Verify it's set
echo $WANDB_API_KEY
```

### 3. Build and Run
```bash
# Build images and start services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

### 4. Access the Application

Open your browser and navigate to:

- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8080
- **API Documentation:** http://localhost:8080/docs

### 5. Stop the Services
```bash
# Stop services (Ctrl+C if running in foreground)

# Or if running in detached mode:
docker-compose down
```

---

## Docker Images

This project uses two Docker images:

### Backend Image
- **Base:** Python 3.9-slim
- **Contains:** FastAPI app, model artifacts
- **Port:** 8080
- **Health check:** `GET /health`

### Frontend Image
- **Base:** Python 3.9-slim
- **Contains:** Streamlit app
- **Port:** 8501
- **Connects to:** Backend at `http://backend:8080`

---

## Environment Variables

### Backend
- `WANDB_API_KEY`: Required for W&B integration (optional if using local model files)
- `PORT`: Default 8080

### Frontend
- `API_URL`: Backend URL (default: `http://backend:8080` in Docker Compose)
- `PORT`: Default 8501

---

## Testing the Deployment

### 1. Test Backend Health
```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"healthy","model_loaded":true}
```

### 2. Test Prediction Endpoint
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
    "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
  }'
```

### 3. Test Frontend

1. Open http://localhost:8501 in browser
2. Enter transaction details
3. Click "Check for Fraud"
4. Verify prediction appears

---

## Viewing Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f

# Last 50 lines
docker-compose logs --tail=50
```

---

## Rebuilding After Changes
```bash
# After changing code
docker-compose down
docker-compose up --build

# Quick rebuild of specific service
docker-compose build backend
docker-compose up backend
```

---


### Screenshots

[Add screenshots of:]
1. Streamlit frontend with prediction
2. W&B dashboard with metrics
3. API documentation (FastAPI /docs)

---

## Production Deployment (Future)

This Docker setup is cloud-ready and can be deployed to:

- **Render:** Free tier, Docker support
- **Railway:** Simple Docker deployment

**Infrastructure is prepared but not currently deployed due to time constraints.**

---

## CI/CD (Configured)

GitHub Actions workflows are configured but not triggered:

- **train-pipeline.yml:** Automated model training
- **deploy.yml:** Whole App deployment 

---

## Additional Resources

- **Docker Documentation:** https://docs.docker.com
- **Docker Compose Reference:** https://docs.docker.com/compose
- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **Streamlit Documentation:** https://docs.streamlit.io
- **W&B Documentation:** https://docs.wandb.ai

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Docker logs: `docker-compose logs`
3. Verify all prerequisites are installed
4. Check GitHub Issues in repository

---

**Built for DS & AI Bootcamp Project**
