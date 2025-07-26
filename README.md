# AntiPhishAI

**A web-based phishing detection system powered by multiple Machine Learning and Deep Learning models.**

---

## 🧠 Overview

AntiPhishAI analyzes URLs and optional email content to detect phishing attempts. The application runs several trained models concurrently and presents prediction outcomes with confidence scores via an intuitive Flask interface.

---

## ⚙️ Key Features

- Submit URLs (and optional email content) via a clean web form
- Extract and compute phishing-relevant features
- Run predictions using:
  - Gradient Boosting
  - K‑Nearest Neighbors (KNN)
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - LSTM (Deep Learning)
- Display labels (**Phishing** or **Legitimate**) and model confidence scores
- Show per-model explanations for interpretability
- Includes Privacy Policy and Terms of Service pages

---

## 🧰 Tech Stack

- **Backend**: Flask (Python)  
- **ML/DL Tools**: scikit-learn, TensorFlow / Keras  
- **Data Handling**: pandas, NumPy  
- **Frontend**: Jinja2-based HTML templates  

---

## 📁 Repository Structure

```

AntiPhishAI/
├── app.py                        # Flask application entry
├── static/
│   └── script/
│       ├── get\_info.py           # Feature extraction logic
│       ├── phishing\_model\_*.pkl  # Trained ML models
│       ├── phishing\_model\_dl\_lstm.h5  # LSTM model
│       └── scaler\_*.pkl          # Scalers for ML models
├── templates/
│   ├── index.html               # Input form page
│   ├── results.html             # Prediction results page
│   ├── privacy.html             # Privacy policy page
│   └── terms.html               # Terms of Service page
├── LICENSE                      # MIT License
├── README.md                    # This file
└── requirements.txt             # Dependencies list

````

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/bsrCh3rif07/AntiPhishAI.git
cd AntiPhishAI
````

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to start.

---

## 📜 How It Works

### 1. Feature Extraction

`get_features_info(domain, email_content)` computes features like URL length, digit ratios, domain age, number of hyperlinks, etc.

### 2. Preprocessing & Modeling

* ML models use a pre-fitted scaler on features, then predict.
* LSTM uses reshaped input `(batch_size, 1, features)` after optional scaling.

### 3. Prediction Logic

A prediction score ≥ 0.01 is labeled **Phishing**, else **Legitimate**.

### 4. Results Display

Each model’s prediction and score are shown, along with a short explanation of how it works.

---

## 🔧 Future Enhancements

* Batch processing via CSV upload interface
* Real-time domain lookup & feature updates
* Dashboard to compare model performance and visualize results
* Integration with email providers for automated scanning

---

## 🕶️ Attribution & License

Licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🧾 TL;DR

AntiPhishAI is a Flask-based phishing classifier that leverages multiple ML and DL models to analyze and flag suspicious URLs and email content—ideal for both learning and demonstration purposes.

