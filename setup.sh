#!/bin/bash

echo "=========================================="
echo "Fraud Detection MLOps - Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download the dataset from Kaggle:"
echo "   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
echo ""
echo "2. Place 'creditcard.csv' in the 'data/' directory"
echo ""
echo "3. Set up Weights & Biases:"
echo "   wandb login"
echo ""
echo "4. Train your first model:"
echo "   python models/train.py --data_path data/creditcard.csv"
echo ""
echo "5. Run locally with Docker Compose:"
echo "   docker-compose up --build"
echo ""
echo "=========================================="