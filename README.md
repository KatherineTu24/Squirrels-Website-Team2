# Tail Trails - NYC Squirrel Census Prediction Project

A web application that uses machine learning to predict squirrel park locations based on observed behaviors, fur colors, and environmental factors. The system is built around a formal ontology (OWL 2.0) that models squirrel sightings and their relationships to NYC parks and geographic areas.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone or download the project** to your local machine

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - `flask==2.3.3` - Web framework
   - `scikit-learn>=1.4.0` - Machine learning library
   - `pandas>=2.0.3` - Data manipulation
   - `rdflib>=7.0.0` - RDF/OWL ontology parsing
   - `joblib>=1.3.2` - Model serialization

## Running the Application

### Setup Steps

1. **Download & unzip the folder** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd path/to/your/folder
   ```
   Replace `path/to/your/folder` with the actual path to your project directory.

3. **Train the machine learning model**:
   ```bash
   py train_model.py
   ```
   This step is **required** before running the application. It will:
   - Load data from `Ontology.tll`
   - Train a Random Forest classifier
   - Save model files to `artifacts/` directory
   - Generate metadata for the web application
   
   You should see output like:
   ```
   Loading TTL file: Ontology.tll
   Loaded 3229 triples from TTL file
   Extracted 283 sightings
   Loaded 283 records from TTL file
   {
     "accuracy": 0.19298245614035087,
     "features_used": [
       "fur_color",
       "location",
       "time_of_day",
       "weather_bucket"
     ],
     "top_feature": {
       "feature": "location",
       "importance": 0.542792253114039
     },
     "n_classes": 17
   }
   ```

### Running the Application (Two Terminal Setup)

**Important**: Make sure you have run `py train_model.py` first to generate the required model files.

The application requires two servers to run properly:

#### **Terminal 1: Start Flask API Server**
```bash
py app.py
```
- This runs the Flask backend API server on **port 5000**
- You should see output like:
  ```
  Loading model...
  Model loaded successfully!
  Accuracy: 0.19298245614035087
  Loading TTL file: Ontology.tll
  Loaded 3523 triples from TTL file
  Extracted 283 sightings
  TTL data loaded successfully!
  Sightings: 283
  Parks: 18
  Starting Flask server...
  * Running on http://127.0.0.1:5000
  ```

#### **Terminal 2: Start Web Server**
Open a **second terminal** in the same directory and run:
```bash
py -m http.server 8000
```
- This runs a simple HTTP server on **port 8000**
- This serves the frontend HTML file

#### **Step 3: Open the Website**
1. Open your web browser
2. Navigate to: **`http://localhost:8000/website_squirrel.html`**
3. The application should load with the vintage paper-themed interface

### Why Two Servers?

- **Port 5000**: Flask API backend handles data requests and ML predictions
- **Port 8000**: Web server serves the frontend HTML file and static assets
- This setup allows the frontend to make API calls to the Flask backend while being served separately

### Project Structure
```
├── app.py                      # Flask backend server
├── website_squirrel.html        # Frontend single-page application
├── train_model.py              # ML model training script
├── ttl_parser.py               # Ontology parsing utilities
├── Ontology.tll                # Main ontology file (TTL format)
├── squirrel-park_merged_clean.csv # Source CSV data
├── requirements.txt             # Python dependencies
└── artifacts/                  # Trained ML model files
    ├── model.pkl               # Scikit-learn Random Forest model
    ├── metadata.json           # Model training metadata
    └── preprocessor.pkl        # Feature preprocessing pipeline
```

### Retraining the Model

To retrain the ML model with updated data:

```bash
py train_model.py
```

This will:
- Load data from `Ontology.tll`
- Train a new Random Forest classifier
- Save updated model files to `artifacts/`

