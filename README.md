# Kalboard 360: Student Performance Prediction & Advising System

A hybrid Artificial Intelligence application that merges **Statistical Machine Learning (ML)** with **Symbolic AI Reasoning** to predict student performance levels and automatically output customized academic advising pathways.

The system is served via a **Python Flask backend** and presented on a modern **frosted glassmorphic Single-Page Dashboard**.

---

## 🚀 Key Features

* **Statistical ML Predictor**: Analyzes student records (demographics, parent satisfaction, and online interactions) to classify performance into **Low (L)**, **Medium (M)**, or **High (H)** bands with confidence scoring.
* **Symbolic AI Engines**:
  1. **BFS Stepper (Search Agent)**: Computes the shortest pathway from a student's risk profile to the target goal (`Pass`) using **Breadth-First Search (BFS)**.
  2. **CSP Study Planner (Backtracking)**: Resolves variables (`StudyHours`, `Tutoring`, `Monitoring`, `Attendance`) satisfying constraints via backtracking search to output a custom study program.
  3. **Logical Inference Logs (Forward Chaining)**: Uses production rules to derive academic facts (e.g. `low_attendance` + `low_grades` $\rightarrow$ `tutoring_required` $\rightarrow$ `success_plan_created`), making the system's reasoning visible.
* **Model Diagnostic Center**: A performance panel displaying comparative accuracies (KNN, Logistic Regression, Random Forest) and interactive feature importances (identifying leading variables like absences, raised hands, and resource visits).
* **Sandbox Profile Loader**: Includes real student samples that can be loaded instantly with one click to test different risk profiles.

---

## 🛠️ Tech Stack
* **Backend**: Python 3.14, Flask, Scikit-learn, Pandas, Joblib
* **Frontend**: HTML5, Vanilla CSS3 (Frosted Glassmorphic layout), Vanilla JavaScript ES6

---

## 📁 Repository Structure
```text
student-performance-advisor/
├── train_and_export.py    # Offline preprocessing, training, and model serialization
├── app.py                 # Core Flask application serving routes and running AI modules
├── student.csv            # Dataset containing academic performance logs
├── requirements.txt       # Python library dependencies
├── models/
│   ├── model.joblib       # Pre-trained Random Forest Classifier
│   ├── scaler.joblib      # Pre-trained StandardScaler scaler
│   ├── metrics.json       # Exported model performance statistics
│   ├── columns.json       # Feature column structural mapping
│   ├── feature_importances.json # RF feature weights
│   └── samples.json       # Sample student profiles for sandbox loading
├── static/
│   ├── css/
│   │   └── style.css      # Dark-themed glassmorphism stylesheet
│   └── js/
│       └── app.js         # AJAX request handling and dynamic DOM updates
└── templates/
    └── index.html         # Main dashboard HTML template
```

---

## 📊 Model Performance Comparison

The training script evaluates three baseline models on a 20% test partition:

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (K-Neighbors)** | 78.4% | 71.9% | 71.6% | 71.9% | 71.5% |
| **Logistic Regression** | 82.0% | 76.0% | 76.0% | 76.0% | 76.0% |
| **Random Forest (Primary)** ⭐ | **91.9%** | **79.2%** | **81.1%** | **79.2%** | **79.2%** |

---

## ⚙️ Installation & Usage Guide

### Prerequisites
Make sure you have **Python 3.8+** and **Git** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/SaifDevAI/student-performance-advisor.git
cd student-performance-advisor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Train & Export the Model
The repository comes pre-bundled with trained models inside the `models/` directory. If you want to retrain the models on the CSV dataset, run:
```bash
python train_and_export.py
```

### 4. Run the Flask Web Application
```bash
python app.py
```

### 5. Access the Web Dashboard
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
