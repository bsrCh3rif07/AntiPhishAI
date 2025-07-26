from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from urllib.parse import urlparse
from datetime import datetime
from static.script.get_info import get_features_info

app = Flask(__name__)

# Paths to model and scaler files and static data
MODEL_PATH = "static/script"
MODELS = {
    "LSTM (Deep Learning)": "phishing_model_dl_lstm.h5",
    "Gradient Boosting": "phishing_model_gb.pkl",
    "K-Nearest Neighbors": "phishing_model_knn.pkl",
    "Logistic Regression": "phishing_model_lr.pkl",
    "Random Forest": "phishing_model_rf.pkl",
    "Support Vector Machine": "phishing_model_svm.pkl"
}
SCALERS = {
    "Gradient Boosting": "scaler_gb.pkl",
    "K-Nearest Neighbors": "scaler_knn.pkl",
    "Logistic Regression": "scaler_lr.pkl",
    "Random Forest": "scaler_rf.pkl",
    "Support Vector Machine": "scaler_svm.pkl"
}

# Function to load models and scalers
def load_models_and_scalers():
    models = {}
    scalers = {}
    for name, filename in MODELS.items():
        model_path = os.path.join(MODEL_PATH, filename)
        if filename.endswith((".h5")):
            models[name] = load_model(model_path, compile=False)
        else:
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)
    for name, filename in SCALERS.items():
        with open(os.path.join(MODEL_PATH, filename), "rb") as f:
            scalers[name] = pickle.load(f)
    return models, scalers


# Load models and scalers globally at the start
models, scalers = load_models_and_scalers()

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')  # Render Privacy Policy page

@app.route('/terms')
def terms():
    return render_template('terms.html')  # Render Terms of Service page


@app.route("/results", methods=["POST"])
def results():
    domain = request.form["domain"]
    email_content = request.form.get("email_content", "")

    # Extract features from URL and email content
    features = get_features_info(domain, email_content)

    # Convert feature values to numerical (float) values
    features_values = [
        float(val) if isinstance(val, (int, float)) 
        else float(val.timestamp()) if isinstance(val, datetime)  # Convert datetime to timestamp
        else 0  # Default for non-numerical values
        for val in features.values()
    ]
    
    # Convert to DataFrame
    df_features = pd.DataFrame([features_values], columns=features.keys())  # Create a DataFrame from the feature values

    results = {}
    
    # Define the feature sets for ML and DL models (Features expected by trained models)
    ml_features = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_qm', 'nb_eq',
        'nb_slash', 'nb_www', 'ratio_digits_url', 'ratio_digits_host', 'tld_in_subdomain',
        'prefix_suffix', 'shortest_word_host', 'longest_words_raw', 'longest_word_path',
        'phish_hints', 'nb_hyperlinks', 'ratio_intHyperlinks', 'empty_title',
        'domain_in_title', 'domain_age', 'google_index', 'page_rank'
    ]

    dl_features = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_qm', 'nb_eq',
        'nb_slash', 'nb_www', 'ratio_digits_url', 'ratio_digits_host', 'tld_in_subdomain',
        'prefix_suffix', 'shortest_word_host', 'longest_words_raw', 'longest_word_path',
        'phish_hints', 'nb_hyperlinks', 'ratio_intHyperlinks', 'empty_title',
        'domain_in_title', 'domain_age', 'google_index', 'page_rank'
    ]

    # Ensure the required features are present, add missing ones with default value of 0
    for feature in ml_features + dl_features:
        if feature not in df_features.columns:
            df_features[feature] = 0  # Add missing feature with default value of 0

    # Create separate DataFrames for ML and DL models
    df_ml = df_features[ml_features]  # Use only ML-specific features
    df_dl = df_features[dl_features]  # Use only DL-specific features

    # Iterate over the models to make predictions
    for model_name, model in models.items():
        if model_name != "LSTM (Deep Learning)":  # Skip LSTM for now and continue with other models
            scaler = scalers.get(model_name)
            
            # Scale or use raw features based on the model
            if scaler:
                # Scale the features for models that require scaling
                scaled_input = scaler.transform(df_ml)  # Use df_ml for ML models
            else:
                # If no scaling is required, just use the raw DataFrame values
                scaled_input = df_ml.values  # Use the values from the DataFrame directly

            # Make predictions with the ML model
            prediction = model.predict(scaled_input)
            prediction_label = "Phishing" if prediction[0] >= 0.01 else "Legitimate"  # Adjust based on the model output format
            
            # Store the results
            results[model_name] = {"prediction": prediction_label, "score": float(prediction[0])}

        else:  # For LSTM (Deep Learning) model
            # Handle scaling and reshaping for LSTM model
            scaler = scalers.get(model_name)
            
            if scaler:
                # Scale the features for models that require scaling
                scaled_input = scaler.transform(df_dl)  # Use df_dl for DL models
            else:
                # If no scaling is required, just use the raw DataFrame values
                scaled_input = df_dl.values  # Use the values from the DataFrame directly
            
            # Reshape the input for LSTM to (batch_size, time_steps, features)
            scaled_input = scaled_input.reshape((scaled_input.shape[0], 1, len(dl_features)))  # Adjust to match the expected input shape
            
            # Make predictions with the LSTM model
            prediction = model.predict(scaled_input)
            prediction_label = "Phishing" if prediction[0] >= 0.01 else "Legitimate"  # Adjust based on the model output format
            
            # Store the results
            results[model_name] = {"prediction": prediction_label, "score": float(prediction[0])}

    explanations = {
        "LSTM (Deep Learning)": "Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to handle sequential data.",
        "Gradient Boosting": "Gradient Boosting builds an ensemble of weak prediction models, usually decision trees, to improve predictions iteratively.",
        "K-Nearest Neighbors": "KNN predicts the class of a sample based on the majority class among its nearest neighbors.",
        "Logistic Regression": "Logistic Regression is a linear model used for binary classification problems.",
        "Random Forest": "Random Forest builds multiple decision trees and combines their predictions for robust results.",
        "Support Vector Machine": "SVM finds the hyperplane that best separates classes in the feature space."
    }

    alert_count = sum([1 for result in results.values() if result["prediction"] == "Phishing"])
    alert_ratio = alert_count / len(results) if len(results) > 0 else 0

    # Render the results page with predictions and explanations
    return render_template(
        "results.html",
        results=results,
        explanations=explanations,
        domain=domain,
        alert_count=alert_count,
        alert_ratio=alert_ratio,
    )

if __name__ == "__main__":
    app.run(debug=True)
