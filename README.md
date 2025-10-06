# Hotel Reservation Cancellation Prediction

A complete MLOps project for predicting hotel reservation cancellations using LightGBM, with automated CI/CD pipelines using Jenkins and deployment on Google Cloud Platform (GCP).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Deployment](#deployment)
- [Web Application](#web-application)
- [MLFlow Integration](#mlflow-integration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline to predict whether a hotel reservation will be canceled. It demonstrates best practices in MLOps including:

- Automated data ingestion from Google Cloud Storage
- Feature engineering and preprocessing
- Model training with hyperparameter tuning
- Experiment tracking with MLFlow
- Containerization with Docker
- CI/CD automation with Jenkins
- Cloud deployment on Google Cloud Run

## ✨ Features

- **Data Ingestion**: Automated data download from GCP buckets
- **Data Preprocessing**: 
  - Handling imbalanced data using SMOTE
  - Feature selection using Random Forest
  - Label encoding for categorical variables
- **Model Training**: 
  - LightGBM classifier with RandomizedSearchCV
  - Hyperparameter tuning
- **Experiment Tracking**: MLFlow for logging parameters, metrics, and artifacts
- **Web Interface**: Flask-based web application for predictions
- **CI/CD Pipeline**: Automated Jenkins pipeline for build and deployment
- **Cloud Deployment**: Containerized deployment on Google Cloud Run

## 🏗 Project Architecture

```
┌─────────────────┐
│  GCP Bucket     │
│  (Raw Data)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│  (Download &    │
│   Split Data)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processing │
│  (Preprocessing,│
│   Balancing,    │
│   Feature Sel.) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  (LightGBM with │
│   MLFlow)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Flask Web App   │
│  (Predictions)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Docker        │
│  (Containerize) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Jenkins       │
│  (CI/CD)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Google Cloud    │
│     Run         │
└─────────────────┘
```

## 🛠 Tech Stack

- **Language**: Python 3.x
- **ML Framework**: LightGBM, scikit-learn
- **Data Processing**: Pandas, NumPy, imbalanced-learn
- **Experiment Tracking**: MLFlow
- **Web Framework**: Flask
- **Cloud Platform**: Google Cloud Platform (GCS, Cloud Run)
- **Containerization**: Docker
- **CI/CD**: Jenkins
- **Version Control**: Git

## 📁 Project Structure

```
Hotel-Reservation-Prediction/
│
├── src/                          # Source code
│   ├── data_injection.py         # Data ingestion from GCP
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── model_training.py         # Model training with MLFlow
│   ├── logger.py                 # Logging configuration
│   └── custom_exception.py       # Custom exception handling
│
├── pipeline/                     # Pipeline scripts
│   └── training_pipeline.py      # Complete training pipeline
│
├── config/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   ├── paths_config.py           # Path configurations
│   └── model_params.py           # Model hyperparameters
│
├── utils/                        # Utility functions
│   └── common_functions.py       # Common helper functions
│
├── templates/                    # Flask HTML templates
│   └── index.html                # Web interface
│
├── static/                       # Static files (CSS, JS)
│
├── artifacts/                    # Generated artifacts
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── models/                   # Trained models
│
├── notebook/                     # Jupyter notebooks
│   └── notebook.ipynb            # Exploratory data analysis
│
├── jenkins/                      # Jenkins configuration
│   └── Dockerfile                # Jenkins Docker setup
│
├── application.py                # Flask application
├── Dockerfile                    # Application Dockerfile
├── Jenkinsfile                   # Jenkins pipeline definition
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account (for data ingestion and deployment)
- Docker (optional, for containerization)
- Jenkins (optional, for CI/CD)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sameeh07/Hotel-Reservation-Prediction.git
   cd Hotel-Reservation-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

## ⚙ Configuration

### 1. Update GCP Configuration

Edit `config/config.yaml`:

```yaml
data_ingestion:
  bucket_name: "your-bucket-name"          # Your GCP bucket name
  bucket_file_name: "HotelReservations.csv"
  train_ratio: 0.8
```

### 2. Set up GCP Credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### 3. Configure Model Parameters

Modify `config/model_params.py` to adjust hyperparameters:

```python
LIGHTGM_PARAMS = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    # ... more parameters
}
```

## 📖 Usage

### Training the Model

Run the complete training pipeline:

```bash
python pipeline/training_pipeline.py
```

This will:
1. Download data from GCP bucket
2. Split data into train/test sets
3. Preprocess and balance the data
4. Select top features
5. Train LightGBM model with hyperparameter tuning
6. Log experiments to MLFlow
7. Save the trained model

### Running Individual Stages

**Data Ingestion Only:**
```bash
python src/data_injection.py
```

**Data Preprocessing Only:**
```bash
python src/data_preprocessing.py
```

**Model Training Only:**
```bash
python src/model_training.py
```

### Running the Web Application

```bash
python application.py
```

The application will be available at `http://localhost:8080`

### Using Docker

**Build the Docker image:**
```bash
docker build -t hotel-reservation-app .
```

**Run the container:**
```bash
docker run -p 8080:8080 hotel-reservation-app
```

## 🔄 Pipeline Stages

### 1. Data Ingestion
- Downloads CSV data from GCP bucket
- Splits data into training (80%) and testing (20%) sets
- Saves raw data to `artifacts/raw/`

### 2. Data Preprocessing
- Handles imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)
- Encodes categorical variables using LabelEncoder
- Selects top 10 features using Random Forest feature importance
- Handles skewness in numerical features
- Saves processed data to `artifacts/processed/`

### 3. Model Training
- Trains LightGBM classifier
- Performs hyperparameter tuning using RandomizedSearchCV
- Evaluates model on test set (accuracy, precision, recall, F1-score)
- Logs all parameters, metrics, and artifacts to MLFlow
- Saves trained model to `artifacts/models/`

## 🌐 Deployment

### Jenkins CI/CD Pipeline

The Jenkinsfile defines a 4-stage pipeline:

1. **Clone Repository**: Pulls latest code from GitHub
2. **Setup Environment**: Creates virtual environment and installs dependencies
3. **Build & Push**: Builds Docker image and pushes to Google Container Registry (GCR)
4. **Deploy**: Deploys container to Google Cloud Run

### Manual Deployment to GCP

1. **Build and push Docker image:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/hotel-reservation-app
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy hotel-reservation-app \
     --image gcr.io/YOUR_PROJECT_ID/hotel-reservation-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## 💻 Web Application

The Flask web application provides a user-friendly interface for making predictions. Users can input reservation details:

- Lead time
- Number of special requests
- Average price per room
- Arrival month and date
- Market segment type
- Number of weeknights and weekend nights
- Type of meal plan
- Room type reserved

The model predicts whether the reservation is likely to be canceled.

## 📊 MLFlow Integration

MLFlow tracks all experiments, including:

- **Parameters**: Model hyperparameters
- **Metrics**: Accuracy, precision, recall, F1-score
- **Artifacts**: 
  - Training and test datasets
  - Trained model files
  - Model parameters

To view MLFlow UI:
```bash
mlflow ui
```

Access at `http://localhost:5000`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sameeh**
- GitHub: [@Sameeh07](https://github.com/Sameeh07)

## 🙏 Acknowledgments

- Dataset: Hotel Reservations Dataset
- LightGBM team for the excellent gradient boosting framework
- MLFlow community for experiment tracking tools
- Google Cloud Platform for hosting infrastructure

---

**Note**: Make sure to replace placeholder values (like `your-bucket-name`, `your-project-name`) with your actual configuration values before running the project.
