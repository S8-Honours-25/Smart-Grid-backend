from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Config
DATASETS_FOLDER = "datasets"
UPLOAD_FOLDER = DATASETS_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure datasets folder exists
os.makedirs(DATASETS_FOLDER, exist_ok=True)

# Global variables
model = None
label_encoder = None
feature_columns = None
current_dataset = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_datasets():
    """Return list of available datasets in the datasets folder"""
    return [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.csv')]

def get_current_dataset():
    """Get the current dataset filename"""
    global current_dataset
    # If no dataset is currently set, use the first available one
    if current_dataset is None:
        available_datasets = get_available_datasets()
        if available_datasets:
            current_dataset = available_datasets[0]
        else:
            # Default if no datasets found
            current_dataset = ""
    return current_dataset

def get_dataset_path():
    """Get the full path to the current dataset"""
    return os.path.join(DATASETS_FOLDER, get_current_dataset())

def load_or_train_model():
    """Load the model from disk or train a new one if it doesn't exist"""
    global model, label_encoder, feature_columns, current_dataset
    
    # Get current dataset filename
    dataset_name = get_current_dataset()
    if not dataset_name:
        print("No dataset available for model training")
        return False
        
    # Generate model filename based on dataset name
    base_name = os.path.splitext(dataset_name)[0]
    model_path = f"{base_name}_rf_model.pkl"
    encoder_path = f"{base_name}_label_encoder.pkl"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        # Load existing model and encoder
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_columns = model_data['features']
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"Model loaded from disk for dataset: {dataset_name}")
        return True
    
    # If model doesn't exist, train a new one
    try:
        # Load data
        dataset_path = get_dataset_path()
        df = pd.read_csv(dataset_path)
        
        # Find target columns (assuming 'stabf' is the categorical target and 'stab' is a numeric measure)
        # This is a simplistic approach - in a real app, you'd want to handle different dataset structures
        if 'stabf' in df.columns:
            target_col = 'stabf'
            numeric_target = 'stab' if 'stab' in df.columns else None
        else:
            # Find potential categorical target columns
            cat_columns = df.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                target_col = cat_columns[-1]  # Just use the last categorical column
            else:
                # If no categorical columns, use the last column
                target_col = df.columns[-1]
            numeric_target = None
        
        print(f"Using {target_col} as target column")
        
        # Encode target variable
        label_encoder = LabelEncoder()
        df[f'{target_col}_encoded'] = label_encoder.fit_transform(df[target_col])
        
        # Prepare features and target
        exclude_cols = [target_col, f'{target_col}_encoded']
        if numeric_target:
            exclude_cols.append(numeric_target)
            
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_columns]
        y = df[f'{target_col}_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        
        # Save the model
        model = rf_classifier
        model_data = {
            'model': model,
            'features': feature_columns
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"Model trained and saved for dataset: {dataset_name}")
        return True
    
    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Return list of available datasets and the current active one"""
    try:
        datasets = get_available_datasets()
        return jsonify({
            'datasets': datasets,
            'current_dataset': get_current_dataset()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/switch-dataset', methods=['POST'])
def switch_dataset():
    """Switch to a different dataset"""
    global current_dataset, model, label_encoder, feature_columns
    
    try:
        data = request.json
        dataset_name = data.get('dataset')
        
        if not dataset_name or dataset_name not in get_available_datasets():
            return jsonify({'error': 'Invalid dataset name'}), 400
        
        # Reset model info
        model = None
        label_encoder = None
        feature_columns = None
        
        # Set new dataset
        current_dataset = dataset_name
        
        # Try to load model for the new dataset
        load_or_train_model()
        
        return jsonify({
            'success': True,
            'message': f'Switched to dataset: {dataset_name}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser submits an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(file_path)
            
            # Switch to the new dataset
            global current_dataset
            current_dataset = filename
            
            # Train model for the new dataset
            success = load_or_train_model()
            
            if not success:
                return jsonify({'error': 'Failed to process dataset'}), 500
                
            return jsonify({
                'success': True,
                'message': 'Dataset uploaded successfully',
                'filename': filename
            })
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset-info', methods=['GET'])
def dataset_info():
    """Return basic information about the dataset"""
    try:
        df = pd.read_csv(get_dataset_path())
        
        # Determine target columns
        if 'stabf' in df.columns:
            target_col = 'stabf'
        else:
            # Find potential categorical target columns
            cat_columns = df.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                target_col = cat_columns[-1]  # Just use the last categorical column
            else:
                # If no categorical columns, use the last column
                target_col = df.columns[-1]
        
        # Get class distribution if target is categorical
        class_distribution = {}
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
            class_distribution = df[target_col].value_counts().to_dict()
        
        result = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'class_distribution': class_distribution,
            'current_dataset': get_current_dataset()
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlations', methods=['GET'])
def correlations():
    """Return correlation data for the dataset"""
    try:
        df = pd.read_csv(get_dataset_path())
        
        # Determine numeric target column (if exists)
        numeric_target = None
        if 'stab' in df.columns:
            numeric_target = 'stab'
        
        # Select numeric columns for correlation
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numeric_columns].corr().to_dict()
        
        # Get correlations with stability if applicable
        target_corr = {}
        if numeric_target and numeric_target in numeric_columns:
            target_corr = dict(df[numeric_columns].corr()[numeric_target].sort_values(ascending=False))
        
        return jsonify({
            'correlation_matrix': correlation_matrix,
            'target_correlations': target_corr,
            'target_column': numeric_target
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Return feature importance from the trained model"""
    global model, feature_columns
    
    if model is None:
        success = load_or_train_model()
        if not success:
            return jsonify({'error': 'Failed to load or train model'}), 500
    
    try:
        importances = model.feature_importances_
        feature_imp = dict(zip(feature_columns, importances.tolist()))
        
        return jsonify({
            'feature_importance': feature_imp
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-performance', methods=['GET'])
def model_performance():
    """Return model performance metrics"""
    global model, label_encoder
    
    if model is None:
        success = load_or_train_model()
        if not success:
            return jsonify({'error': 'Failed to load or train model'}), 500
    
    try:
        # Load data
        df = pd.read_csv(get_dataset_path())
        
        # Determine target column
        if 'stabf' in df.columns:
            target_col = 'stabf'
        else:
            # Find potential categorical target columns
            cat_columns = df.select_dtypes(include=['object']).columns
            if len(cat_columns) > 0:
                target_col = cat_columns[-1]  # Just use the last categorical column
            else:
                # If no categorical columns, use the last column
                target_col = df.columns[-1]
        
        # Encode target if needed
        if f'{target_col}_encoded' not in df.columns:
            df[f'{target_col}_encoded'] = label_encoder.transform(df[target_col])
        
        X = df[feature_columns]
        y = df[f'{target_col}_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        
        # Convert classification report to dictionary
        report = classification_report(y_test, y_pred, 
                                     target_names=label_encoder.classes_,
                                     output_dict=True)
        
        return jsonify({
            'accuracy': accuracy,
            'classification_report': report
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions based on input features"""
    global model, label_encoder, feature_columns
    
    if model is None:
        success = load_or_train_model()
        if not success:
            return jsonify({'error': 'Failed to load or train model'}), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Create dataframe from input data
        input_df = pd.DataFrame(data, index=[0])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                return jsonify({'error': f'Missing feature: {col}'}), 400
        
        # Make prediction
        prediction_idx = model.predict(input_df[feature_columns])
        prediction_proba = model.predict_proba(input_df[feature_columns])
        
        # Convert prediction to original class label
        prediction_class = label_encoder.inverse_transform(prediction_idx)[0]
        
        # Get class probabilities
        class_probabilities = dict(zip(label_encoder.classes_, prediction_proba[0].tolist()))
        
        return jsonify({
            'prediction': prediction_class,
            'probabilities': class_probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """Return a sample of the dataset"""
    try:
        df = pd.read_csv(get_dataset_path())
        
        # Sample limit (default to 10 if not specified)
        limit = request.args.get('limit', default=10, type=int)
        
        # Get sample
        sample = df.sample(min(limit, len(df))).to_dict(orient='records')
        
        return jsonify({
            'sample': sample
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Add this to your app.py file, before the if __name__ == '__main__': line

@app.route('/')
def home():
    """Homepage to show API is running"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            code {
                background-color: #f1f1f1;
                padding: 2px 5px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <h1>ML Model API</h1>
        <div class="card">
            <h2>✅ API is running</h2>
            <p>The Machine Learning Model API is up and running. You can use the following endpoints:</p>
            <ul>
                <li><code>/api/datasets</code> - List available datasets</li>
                <li><code>/api/dataset-info</code> - Get information about current dataset</li>
                <li><code>/api/sample-data</code> - Get sample data from current dataset</li>
                <li><code>/api/model-performance</code> - Get model performance metrics</li>
                <li><code>/api/feature-importance</code> - Get feature importance from the model</li>
                <li><code>/api/correlations</code> - Get correlations from the dataset</li>
                <li><code>/api/predict</code> - Make predictions (POST request)</li>
            </ul>
        </div>
        <div class="card">
            <h3>API Status</h3>
            <p>Current date: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Datasets available: """ + str(len(get_available_datasets())) + """</p>
            <p>Current dataset: """ + (get_current_dataset() or "None") + """</p>
            <p>Model loaded: """ + ("Yes" if model is not None else "No") + """</p>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()
    app.run(debug=True, port=5000)