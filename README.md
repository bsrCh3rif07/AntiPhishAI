Sure! Here's the full `README.md` code as plain text — you can copy-paste it directly into your project directory:

```markdown
# Phishing Detection Web Application using Machine Learning and Deep Learning

## 🧠 Objective

This project aims to detect phishing URLs using various Machine Learning (ML) and Deep Learning (DL) algorithms. It provides a web interface where users can submit a domain or email content. The app extracts features from the input and predicts whether it's legitimate or phishing using multiple trained models.

---

## 🚀 Features

- 🌐 Web-based interface built with Flask  
- 🔍 Feature extraction from domain and email content  
- 🤖 Multi-model predictions using:
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - LSTM (Deep Learning)
- 📊 Displays prediction results and model confidence scores  
- 🔐 Includes privacy and terms policy routes  

---

## 🧰 Tech Stack

- **Backend:** Python (Flask)  
- **Machine Learning:** Scikit-learn, TensorFlow Keras  
- **Data Manipulation:** Pandas, NumPy  
- **Frontend:** HTML (Jinja2 templates)  
- **Deployment-ready:** Localhost or any WSGI-compatible environment  

---

## 🗂️ Project Structure

```

.
├── app.py                         # Main Flask application
├── static/
│   └── script/
│       ├── get\_info.py           # Feature extraction logic
│       ├── phishing\_model\_*.pkl  # ML models
│       ├── phishing\_model\_dl\_lstm.h5  # LSTM model
│       └── scaler\_*.pkl          # Feature scalers
├── templates/
│   ├── index.html                # Input form page
│   ├── results.html              # Result display
│   ├── privacy.html              # Privacy policy
│   └── terms.html                # Terms of service

````

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/phishing-detector-flask.git
cd phishing-detector-flask
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure to include:

* Flask
* numpy
* pandas
* scikit-learn
* tensorflow
* any other dependencies your `get_info.py` uses

### 4. Start the Flask App

```bash
python app.py
```

Then visit `http://127.0.0.1:5000` in your browser.

---

## 🧪 Model Training (Optional)

This repo only includes inference logic. If you want to train the models yourself, prepare a dataset of phishing and legitimate URLs, extract features (like those in `get_features_info()`), and train each algorithm separately, saving:

* Models to `.pkl` or `.h5`
* Corresponding `StandardScaler` objects

---

## 📌 Notes

* The LSTM model expects input reshaped to `(batch_size, time_steps=1, features)`.
* For consistency, the same feature set and order must be used for training and inference.
* The app uses a soft threshold of `0.01` to classify results. Adjust as needed based on your model calibration.

---

## 📜 License

This project is licensed under the MIT License.
