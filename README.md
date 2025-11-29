# Credit Card Fraud Detection - MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A complete end-to-end MLOps pipeline for credit card fraud detection, demonstrating reproducible ML workflows, automated training, model registry, and production deployment.

## 🎯 Project Overview

This project implements a production-grade machine learning system for detecting fraudulent credit card transactions. It showcases:

- ✅ Automated training pipeline with experiment tracking
- ✅ Model versioning and registry (Weights & Biases)
- ✅ CI/CD automation (GitHub Actions)
- ✅ Containerized microservices (Docker)
- ✅ Production deployment (Google Cloud Run)
- ✅ RESTful API for inference (FastAPI)
- ✅ Interactive web interface (Streamlit)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Repository                          │
│  - Training Code    - Backend API    - Frontend UI          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               GitHub Actions (CI/CD)                         │
│  - Model Training   - Testing   - Deployment                │
└──────────┬───────────────────────┬──────────────────────────┘
           │                       │
           ▼                       ▼
┌──────────────────────┐   ┌─────────────────────────────────┐
│  Weights & Biases    │   │    Google Cloud Run             │
│  - Experiments       │   │  ┌──────────────────────────┐   │
│  - Model Registry    │   │  │  FastAPI Backend         │   │
│  - Artifacts         │   │  │  (Model Inference)       │   │
└──────────────────────┘   │  └──────────┬───────────────┘   │
                           │             │                   │
                           │  ┌──────────▼───────────────┐   │
                           │  │  Streamlit Frontend      │   │
                           │  │  (User Interface)        │   │
                           │  └──────────────────────────┘   │
                           └─────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Weights & Biases account
- Google Cloud Platform account (for deployment)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the `data/` directory.

### 4. Set Up Weights & Biases

```bash
# Login to W&B
wandb login

# Set API key as environment variable
export WANDB_API_KEY=your_api_key_here
```

### 5. Train Model

```bash
python models/train.py --data_path data/creditcard.csv
```

### 6. Run Locally with Docker Compose

```bash
# Build and start services
docker-compose up --build

# Access services:
# - Backend API: http://localhost:8080
# - Frontend UI: http://localhost:8501
```

## 📊 Model Details

### Algorithm
- **Model Type:** LightGBM (Gradient Boosting)
- **Framework:** scikit-learn + LightGBM
- **Training Time:** ~2-3 minutes on CPU

### Features
- **Input Features:** 30 (Time, V1-V28, Amount)
- **Target:** Binary (0 = Legitimate, 1 = Fraud)
- **Class Imbalance:** Handled via `scale_pos_weight` parameter

### Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 95% |
| Recall | 88% |
| F1-Score | 91% |
| ROC-AUC | 98% |

*Note: Metrics from baseline model. Actual values will vary.*

## 🔄 MLOps Pipeline

### 1. Training Pipeline

**Trigger:** Push to `main` branch or manual dispatch

**Steps:**
1. Load and validate data
2. Preprocess features (scaling)
3. Train LightGBM model
4. Evaluate performance
5. Log metrics to W&B
6. Save model to W&B registry
7. Trigger deployment if metrics pass threshold

### 2. Model Registry

**Platform:** Weights & Biases

**Versioning Strategy:**
- Each training run creates a new model version
- Models are tagged with aliases: `latest`, `production`, `staging`
- Metadata includes metrics, hyperparameters, and training date

### 3. Deployment Pipeline

**Backend Deployment:**
- Containerized FastAPI application
- Automatically pulls latest `production` model from W&B
- Deployed to Google Cloud Run
- Auto-scaling based on traffic

**Frontend Deployment:**
- Streamlit web application
- Connects to backend via REST API
- Deployed to Google Cloud Run

## 🛠️ Technology Stack

### Machine Learning
- **Framework:** scikit-learn, LightGBM
- **Experiment Tracking:** Weights & Biases
- **Model Registry:** W&B Artifacts

### Backend
- **API Framework:** FastAPI
- **Server:** Uvicorn
- **Validation:** Pydantic

### Frontend
- **Framework:** Streamlit
- **Visualization:** Plotly
- **HTTP Client:** Requests

### DevOps
- **CI/CD:** GitHub Actions
- **Containerization:** Docker
- **Deployment:** Google Cloud Run
- **Version Control:** Git/GitHub

## 📁 Project Structure

```
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       ├── train-pipeline.yml      # Training automation
│       ├── deploy-backend.yml      # Backend deployment
│       └── deploy-frontend.yml     # Frontend deployment
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                      # Streamlit application
│   ├── requirements.txt
│   └── Dockerfile
├── models/
│   ├── train.py                    # Training script
│   ├── config.yaml                 # Model configuration
│   └── artifacts/                  # Saved models (local)
├── data/
│   └── creditcard.csv             # Dataset (not in repo)
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── docker-compose.yml              # Local development
├── requirements.txt                # Python dependencies
└── README.md
```

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model-info
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "Time": 0.0,
  "V1": -1.359807,
  "V2": -0.072781,
  ...
  "V28": -0.021053,
  "Amount": 149.62
}
```

### Batch Prediction
```bash
POST /batch-predict
Content-Type: application/json

{
  "transactions": [...]
}
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Test API locally
curl http://localhost:8080/health

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

## 🚀 Deployment

### Deploy to Google Cloud Run

1. **Set up GCP credentials:**
```bash
# Create service account and download key
gcloud iam service-accounts create fraud-detection-sa
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:fraud-detection-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
```

2. **Add GitHub Secrets:**
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_SA_KEY`: Service account JSON key
- `WANDB_API_KEY`: Your W&B API key

3. **Push to main branch:**
```bash
git push origin main
```

GitHub Actions will automatically:
- Train the model
- Deploy backend to Cloud Run
- Deploy frontend to Cloud Run

## 📈 Monitoring

- **Experiment Tracking:** View training runs at https://wandb.ai/your-username/fraud-detection-mlops
- **Model Registry:** Browse model versions in W&B
- **Deployment Logs:** Check Cloud Run logs in GCP Console
- **Application Metrics:** Monitor request latency, error rates in Cloud Run

## 🔐 Security Considerations

- API keys stored as GitHub Secrets
- Model artifacts stored in W&B (not in repo)
- Cloud Run services use managed identities
- Input validation via Pydantic schemas
- Rate limiting (can be added via Cloud Run)

## 🎓 Learning Outcomes

This project demonstrates:

1. **Version Control:** Git for code, W&B for data/models
2. **Experiment Tracking:** Systematic logging of metrics and hyperparameters
3. **CI/CD:** Automated testing, training, and deployment
4. **Model Registry:** Centralized model versioning and governance
5. **Containerization:** Reproducible environments via Docker
6. **Microservices:** Separation of concerns (backend/frontend)
7. **Cloud Deployment:** Scalable serverless deployment
8. **Monitoring:** Observability into model performance

## 📝 Design Decisions

### Why LightGBM?
- Fast training on CPU
- Handles imbalanced data well via `scale_pos_weight`
- Excellent performance on tabular data
- Built-in feature importance

### Why Weights & Biases?
- Integrated experiment tracking + model registry
- Easy artifact management
- Great visualization capabilities
- Simple API for logging

### Why FastAPI?
- Modern, fast framework
- Automatic API documentation
- Async support for better performance
- Built-in validation via Pydantic

### Why Google Cloud Run?
- Serverless (no infrastructure management)
- Auto-scaling based on traffic
- Cost-effective (pay-per-use)
- Easy CI/CD integration

## 🐛 Troubleshooting

### Issue: Model not loading in backend
**Solution:** Ensure `WANDB_API_KEY` is set and model exists in W&B registry

### Issue: Frontend can't connect to backend
**Solution:** Check `API_URL` environment variable in frontend

### Issue: Training fails with memory error
**Solution:** Reduce dataset size or use cloud compute with more RAM

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👏 Acknowledgments

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- MLOps Inspiration: Chip Huyen's "Designing Machine Learning Systems"
- Tools: Weights & Biases, FastAPI, Streamlit, Google Cloud

## 📧 Contact

For questions or feedback:
- GitHub Issues: [Create an issue](https://github.com/yourusername/fraud-detection-mlops/issues)
- Email: your.email@example.com

---

**Built with ❤️ as part of AI/ML Bootcamp Final Project**