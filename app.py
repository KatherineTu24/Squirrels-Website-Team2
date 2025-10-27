from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import json
from ttl_parser import parse_ttl_file, convert_ttl_to_csv_format, get_statistics_from_ttl

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and metadata
MODEL_PATH = "artifacts/model.pkl"
META_PATH = "artifacts/metadata.json"

model = None
metadata = None
ttl_data = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return False
    
    if not os.path.exists(META_PATH):
        print(f"Metadata file not found: {META_PATH}")
        return False
    
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        
        with open(META_PATH, 'r') as f:
            metadata = json.load(f)
            print(f"Accuracy: {metadata.get('accuracy', 'Unknown')}")
            
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def load_ttl_data():
    """Load TTL data"""
    global ttl_data
    
    ttl_file = "Ontology.tll"
    if not os.path.exists(ttl_file):
        raise FileNotFoundError(f"TTL file not found: {ttl_file}")
    
    ttl_data = parse_ttl_file(ttl_file)
    print(f"TTL data loaded successfully!")
    print(f"Sightings: {len(ttl_data['sightings'])}")
    print(f"Parks: {len(ttl_data['parks'])}")
    
    return ttl_data

def predict_park_probabilities(fur_color='Gray', location='Ground Plane', time_of_day='afternoon', weather_bucket='clear'):
    """Predict park probabilities using the loaded model"""
    if model is None or metadata is None:
        print("Warning: Model or metadata not loaded")
        return {}
    
    try:
        # Get the exact feature names used in training from metadata
        features_used = metadata.get('features_used', ['fur_color', 'location', 'time_of_day', 'weather_bucket'])
        
        # Apply the same preprocessing as in training (matching train_model.py)
        # Fur color: .str.title() matching line 128 in train_model.py
        fur_color_clean = str(fur_color).strip().title() if fur_color else "Unknown"
        
        # Location: .str.title() matching line 132 in train_model.py  
        location_clean = str(location).strip().title() if location else "Unknown"
        
        # Time of day: lowercase matching lines 74-85 in train_model.py
        time_of_day_clean = str(time_of_day).strip().lower() if time_of_day else "unknown"
        
        # Weather bucket: lowercase matching lines 95-117 in train_model.py
        weather_bucket_clean = str(weather_bucket).strip().lower() if weather_bucket else "unknown"
        
        # Create feature vector using the EXACT column names from training
        # These should match the features_used from metadata
        features = {}
        for feature_name in features_used:
            if feature_name == 'fur_color':
                features[feature_name] = fur_color_clean
            elif feature_name == 'location':
                features[feature_name] = location_clean
            elif feature_name == 'time_of_day':
                features[feature_name] = time_of_day_clean
            elif feature_name == 'weather_bucket':
                features[feature_name] = weather_bucket_clean
        
        # Convert to DataFrame - column names must match training exactly
        df = pd.DataFrame([features])
        
        print(f"Making prediction with features: {features}")
        print(f"DataFrame columns: {list(df.columns)}")
        
        # The model is a pipeline that includes OneHotEncoder preprocessing
        # This should now work correctly with proper feature names and format
        probabilities = model.predict_proba(df)[0]
        
        # Get park names from metadata
        park_names = metadata.get('parks', [])
        
        print(f"Model returned {len(probabilities)} probabilities for {len(park_names)} parks")
        
        # Handle case where model returns different number of classes than expected
        if len(probabilities) != len(park_names):
            print(f"Warning: Model returned {len(probabilities)} probabilities but metadata has {len(park_names)} parks")
            # Use the minimum to avoid index errors
            min_len = min(len(probabilities), len(park_names))
            park_names = park_names[:min_len]
            probabilities = probabilities[:min_len]
        
        # Create results dictionary with probabilities
        results = {}
        for i, park in enumerate(park_names):
            results[park] = float(probabilities[i])
        
        print(f"Prediction results: {results}")
        return results
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print(f"Feature vector was: {features if 'features' in locals() else 'Not created'}")
        import traceback
        traceback.print_exc()
        return {}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'metadata_loaded': metadata is not None,
        'ttl_data_loaded': ttl_data is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for getting park predictions"""
    try:
        data = request.get_json()
        
        # Extract parameters with defaults
        fur_color = data.get('fur_color', 'Gray')
        location = data.get('location', 'Ground Plane')
        time_of_day = data.get('time_of_day', 'afternoon')
        weather_bucket = data.get('weather_bucket', 'clear')
        
        # Get predictions
        probabilities = predict_park_probabilities(
            fur_color=fur_color,
            location=location,
            time_of_day=time_of_day,
            weather_bucket=weather_bucket
        )
        
        # Sort by probability and get top 5
        sorted_parks = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_5_parks = dict(sorted_parks[:5])
        
        return jsonify({
            'success': True,
            'probabilities': top_5_parks,
            'all_probabilities': probabilities,
            'input_features': {
                'fur_color': fur_color,
                'location': location,
                'time_of_day': time_of_day,
                'weather_bucket': weather_bucket
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data/csv-format', methods=['GET'])
def get_csv_format_data():
    """Get TTL data converted to CSV format for ML compatibility"""
    if ttl_data is None:
        return jsonify({'error': 'TTL data not loaded'}), 500
    
    csv_data = convert_ttl_to_csv_format(ttl_data)
    return jsonify({
        'data': csv_data,
        'total': len(csv_data)
    })

@app.route('/data/park-details', methods=['GET'])
def get_park_details():
    """Get detailed park information including color distribution and activity patterns"""
    if ttl_data is None:
        return jsonify({'error': 'TTL data not loaded'}), 500
    
    # Get optional color filter from query parameters
    selected_color = request.args.get('color', '').lower()
    
    park_details = []
    
    for park_id, park_info in ttl_data['parks'].items():
        park_sightings = [s for s in ttl_data['sightings'] if s['park_id'] == park_id]
        
        # Color distribution - initialize all colors to 0
        all_colors = ['gray', 'black', 'cinnamon']
        color_counts = {color: 0 for color in all_colors}
        activity_counts = {}
        total_sightings = len(park_sightings)
        
        for sighting in park_sightings:
            # Count colors
            color = sighting.get('fur_color', 'unknown')
            if color in color_counts:
                color_counts[color] += 1
            
            # Count activities from TTL data
            activities = sighting.get('activities', ['Foraging'])
            if isinstance(activities, list):
                # Handle multiple activities
                for activity in activities:
                    activity_counts[activity] = activity_counts.get(activity, 0) + 1
            else:
                # Handle single activity string
                activity_counts[activities] = activity_counts.get(activities, 0) + 1
        
        # Calculate percentages - all colors will be shown
        color_percentages = {}
        for color in all_colors:
            color_percentages[color] = round((color_counts[color] / total_sightings * 100), 1) if total_sightings > 0 else 0
        
        activity_percentages = {}
        for activity, count in activity_counts.items():
            activity_percentages[activity] = round((count / total_sightings * 100), 1) if total_sightings > 0 else 0
        
        # If a specific color is selected, filter activities to only show activities for that color
        filtered_activity_distribution = activity_counts
        filtered_activity_percentages = activity_percentages
        
        if selected_color and selected_color in all_colors:
            # Filter park sightings to only include the selected color
            color_specific_sightings = [s for s in park_sightings if s.get('fur_color', '').lower() == selected_color]
            color_specific_activity_counts = {}
            
            for sighting in color_specific_sightings:
                activities = sighting.get('activities', ['Foraging'])
                if isinstance(activities, list):
                    # Handle multiple activities
                    for activity in activities:
                        color_specific_activity_counts[activity] = color_specific_activity_counts.get(activity, 0) + 1
                else:
                    # Handle single activity string
                    color_specific_activity_counts[activities] = color_specific_activity_counts.get(activities, 0) + 1
            
            # Calculate percentages based on color-specific sightings
            color_total_sightings = len(color_specific_sightings)
            filtered_activity_distribution = color_specific_activity_counts
            filtered_activity_percentages = {}
            for activity, count in color_specific_activity_counts.items():
                filtered_activity_percentages[activity] = round((count / color_total_sightings * 100), 1) if color_total_sightings > 0 else 0
        
        park_details.append({
            'id': park_id,
            'name': park_info['name'],
            'total_sightings': total_sightings,
            'color_distribution': color_counts,
            'color_percentages': color_percentages,
            'activity_distribution': filtered_activity_distribution,
            'activity_percentages': filtered_activity_percentages,
            'most_common_color': max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else 'unknown'
        })
    
    # Sort by total sightings
    park_details.sort(key=lambda x: x['total_sightings'], reverse=True)
    
    return jsonify({
        'parks': park_details,
        'total': len(park_details)
    })

@app.route('/data/education-hq', methods=['GET'])
def get_education_hq_data():
    """Get comprehensive data for Education HQ page"""
    if ttl_data is None:
        return jsonify({'error': 'TTL data not loaded'}), 500
    
    # Get basic statistics
    stats = get_statistics_from_ttl(ttl_data)
    
    # Calculate detailed park statistics
    park_stats = []
    for park_id, park_info in ttl_data['parks'].items():
        park_sightings = [s for s in ttl_data['sightings'] if s['park_id'] == park_id]
        
        # Count by color (count unique sightings, not squirrel counts)
        color_counts = {}
        total_sightings = 0
        for sighting in park_sightings:
            color = sighting['fur_color'] or 'unknown'
            # Count unique sightings, not the number of squirrels in each sighting
            color_counts[color] = color_counts.get(color, 0) + 1
            total_sightings += 1
        
        # Calculate percentages
        color_percentages = {}
        for color, count in color_counts.items():
            color_percentages[color] = round((count / total_sightings * 100), 1) if total_sightings > 0 else 0
        
        park_stats.append({
            'name': park_info['name'],
            'id': park_id,
            'total_sightings': total_sightings,
            'unique_observations': len(park_sightings),
            'color_distribution': color_counts,
            'color_percentages': color_percentages,
            'most_common_color': max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else 'unknown'
        })
    
    # Sort parks by total sightings
    park_stats.sort(key=lambda x: x['total_sightings'], reverse=True)
    
    # Calculate overall color distribution
    overall_color_counts = stats['color_distribution']
    total_observations = stats['total_observations']
    overall_color_percentages = {}
    for color, count in overall_color_counts.items():
        overall_color_percentages[color] = round((count / total_observations * 100), 1) if total_observations > 0 else 0
    
    return jsonify({
        'total_observations': total_observations,
        'unique_parks': len(ttl_data['parks']),
        'unique_areas': len(ttl_data['areas']),
        'overall_color_distribution': overall_color_counts,
        'overall_color_percentages': overall_color_percentages,
        'season_distribution': stats['season_distribution'],
        'park_statistics': park_stats,
        'top_parks': park_stats[:10],  # Top 10 parks by sightings
        'model_accuracy': metadata['accuracy'] if metadata else 0.193
    })

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    
    print("Loading TTL file: Ontology.tll")
    try:
        load_ttl_data()
        print("TTL data loaded successfully!")
    except Exception as e:
        print(f"Error loading TTL data: {e}")
        print("Server will start but data endpoints will return errors")
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
