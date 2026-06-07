import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from collections import deque
import joblib

app = Flask(__name__, template_folder='templates', static_folder='static')

# Set paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
COLUMNS_PATH = os.path.join(MODELS_DIR, "columns.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
IMPORTANCES_PATH = os.path.join(MODELS_DIR, "feature_importances.json")
SAMPLES_PATH = os.path.join(MODELS_DIR, "samples.json")

# Load Serialized Assets on Startup (instant, low CPU)
print("Loading model and scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLUMNS_PATH, "r") as f:
    feature_columns = json.load(f)

with open(METRICS_PATH, "r") as f:
    metrics_data = json.load(f)

with open(IMPORTANCES_PATH, "r") as f:
    importances_data = json.load(f)

with open(SAMPLES_PATH, "r") as f:
    samples_data = json.load(f)

numeric_columns = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']

# ==========================================
# Symbolic AI 1: BFS Advising Path Graph
# ==========================================
academic_graph = {
    "High Risk": ["Assign Tutor", "Counseling Session"],
    "Assign Tutor": ["Personalized Study Plan"],
    "Counseling Session": ["Personalized Study Plan"],
    "Personalized Study Plan": ["Weekly Monitoring"],
    "Weekly Monitoring": ["Pass"],
    "Low Risk": ["Regular Monitoring"],
    "Regular Monitoring": ["Pass"]
}

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph.get(node, []):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None

def generate_recommendation(prediction):
    # Mapping Class: L (0) -> High Risk, M (1) & H (2) -> Low Risk (or custom logic)
    # Class L = 0 (Low Performance) -> High Risk
    # Class M/H = 1 or 2 (Medium/High Performance) -> Low Risk
    if prediction == 0:
        start_state = "High Risk"
    else:
        start_state = "Low Risk"
    
    path = bfs(academic_graph, start_state, "Pass")
    return path

# ==========================================
# Symbolic AI 2: CSP Solver (Backtracking)
# ==========================================
variables = ["StudyHours", "Attendance", "Tutoring", "Monitoring"]
domains = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": ["Low", "Medium", "High"],
    "Tutoring": ["Yes", "No"],
    "Monitoring": ["Yes", "No"]
}

def is_valid(assignment):
    if "StudyHours" in assignment:
        if assignment["StudyHours"] < 3:
            return False
    if "Tutoring" in assignment:
        if assignment["Tutoring"] != "Yes":
            return False
    if "Attendance" in assignment:
        if assignment["Attendance"] == "Low":
            return False
    if "Monitoring" in assignment:
        if assignment["Monitoring"] != "Yes":
            return False
    return True

def backtrack(assignment):
    if len(assignment) == len(variables):
        return assignment
    
    unassigned = [v for v in variables if v not in assignment][0]
    for value in domains[unassigned]:
        assignment[unassigned] = value
        if is_valid(assignment):
            result = backtrack(assignment)
            if result is not None:
                return result
        del assignment[unassigned]
    return None

# ==========================================
# Symbolic AI 3: Forward Chaining Rules
# ==========================================
rules = [
    ({"low_attendance"}, "high_risk"),
    ({"low_grades"}, "high_risk"),
    ({"high_risk"}, "tutoring_required"),
    ({"tutoring_required"}, "study_plan_required"),
    ({"study_plan_required"}, "advisor_meeting"),
    ({"high_study_time"}, "improved_performance"),
    ({"good_attendance"}, "improved_performance"),
    ({"improved_performance"}, "pass_likely"),
    ({"advisor_meeting"}, "weekly_monitoring"),
    ({"weekly_monitoring"}, "academic_support_active"),
    ({"academic_support_active"}, "success_plan_created"),
    ({"pass_likely"}, "graduation_path")
]

def forward_chaining(initial_facts, rules_list):
    facts = set(initial_facts)
    inference_trace = []
    changed = True
    while changed:
        changed = False
        for conditions, conclusion in rules_list:
            if conditions.issubset(facts) and conclusion not in facts:
                facts.add(conclusion)
                inference_trace.append({
                    "conditions": list(conditions),
                    "conclusion": conclusion
                })
                changed = True
    return list(facts), inference_trace

def create_initial_facts(prediction):
    if prediction == 0:  # Low
        return {"low_attendance", "low_grades"}
    else:  # Medium or High
        return {"good_attendance", "high_study_time"}

# ==========================================
# Flask Routes
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics_api():
    return jsonify({
        "metrics": metrics_data,
        "importances": importances_data
    })

@app.route('/api/samples', methods=['GET'])
def get_samples_api():
    return jsonify(samples_data)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        raw_data = request.json
        
        # 1. Map Raw Frontend inputs to the One-Hot / Binary format expected by scaler and model
        record = {col: 0 for col in feature_columns}
        
        # Numeric scores
        record["raisedhands"] = int(raw_data.get("raisedhands", 0))
        record["VisITedResources"] = int(raw_data.get("VisITedResources", 0))
        record["AnnouncementsView"] = int(raw_data.get("AnnouncementsView", 0))
        record["Discussion"] = int(raw_data.get("Discussion", 0))
        
        # GradeID and StageID mappings
        record["GradeID"] = int(str(raw_data.get("GradeID", "2")).replace('G-', ''))
        
        stage_map = {'lowerlevel': 0, 'MiddleSchool': 1, 'HighSchool': 2}
        record["StageID"] = stage_map.get(raw_data.get("StageID"), 0)
        
        # Binary encoded flags
        record["gender_encoded"] = 1 if raw_data.get("gender") == "M" else 0
        record["Semester_encoded"] = 1 if raw_data.get("Semester") == "S" else 0
        record["Relation_encoded"] = 1 if raw_data.get("Relation") == "Mum" else 0
        record["ParentAnsweringSurvey_encoded"] = 1 if raw_data.get("ParentAnsweringSurvey") == "Yes" else 0
        record["ParentschoolSatisfaction_encoded"] = 1 if raw_data.get("ParentschoolSatisfaction") == "Good" else 0
        record["StudentAbsenceDays_encoded"] = 1 if raw_data.get("StudentAbsenceDays") == "Under-7" else 0
        
        # One-hot encoded SectionID
        sec_id = raw_data.get("SectionID", "A")
        sec_col = f"SectionID_{sec_id}"
        if sec_col in record:
            record[sec_col] = 1
            
        # One-hot encoded Topic
        topic_id = raw_data.get("Topic", "IT")
        topic_col = f"Topic_{topic_id}"
        if topic_col in record:
            record[topic_col] = 1
            
        # 2. Build single-row DataFrame with exact columns and order
        student_df = pd.DataFrame([record])
        student_df = student_df[feature_columns]
        
        # 3. Scale numeric features
        student_df_scaled = student_df.copy()
        student_df_scaled[numeric_columns] = scaler.transform(student_df[numeric_columns])
        
        # 4. Predict
        prediction = int(model.predict(student_df_scaled)[0])  # 0, 1, 2
        proba_list = model.predict_proba(student_df_scaled)[0]
        confidence = float(proba_list[prediction])
        
        # Map class indexes back to labels for display
        class_mapping = {0: "Low Performance (L)", 1: "Medium Performance (M)", 2: "High Performance (H)"}
        predicted_class_label = class_mapping.get(prediction)
        
        # 5. BFS Advising Search
        advising_path = generate_recommendation(prediction)
        
        # 6. CSP Study Plan
        csp_plan = backtrack({})
        
        # 7. Forward Chaining Facts & Trace
        initial_facts = create_initial_facts(prediction)
        final_facts, trace = forward_chaining(initial_facts, rules)
        
        # Return response
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "prediction_label": predicted_class_label,
            "confidence": confidence,
            "advising_path": advising_path,
            "csp_plan": csp_plan,
            "initial_facts": list(initial_facts),
            "final_facts": final_facts,
            "inference_trace": trace
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    print("Flask server running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
