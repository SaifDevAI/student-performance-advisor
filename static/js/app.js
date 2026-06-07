document.addEventListener('DOMContentLoaded', () => {
    // 1. Slider values synchronization
    const sliders = [
        { id: 'raisedhands', valId: 'raisedhands-val' },
        { id: 'VisITedResources', valId: 'visitedresources-val' },
        { id: 'AnnouncementsView', valId: 'announcements-val' },
        { id: 'Discussion', valId: 'discussion-val' }
    ];

    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        if (inputEl && valEl) {
            inputEl.addEventListener('input', (e) => {
                valEl.textContent = e.target.value;
            });
        }
    });

    // 2. Load API Static Metrics and Samples
    loadModelMetrics();
    loadSampleProfiles();

    // 3. Form Submission Interceptor
    const form = document.getElementById('student-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        analyzeStudent();
    });
});

// Tab Switcher Logic
function switchTab(tabId) {
    // Deactivate all tabs
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));

    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Activate selected
    document.getElementById(tabId).classList.add('active');
    
    // Find matching button
    buttons.forEach(btn => {
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });
}

// Load Model Evaluation Metrics & Feature Importances
function loadModelMetrics() {
    fetch('/api/metrics?t=' + Date.now())
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('metrics-table-body');
            tbody.innerHTML = ''; // Clear loading spinner
            
            // Map keys to readable titles
            const namesMap = {
                "KNN": "K-Nearest Neighbors (Baseline)",
                "LogisticRegression": "Logistic Regression",
                "RandomForest": "Random Forest Classifier"
            };

            // Populate Supervised Table
            Object.keys(data.metrics).forEach(key => {
                const modelData = data.metrics[key];
                const tr = document.createElement('tr');
                if (key === 'RandomForest') {
                    tr.classList.add('best-model-row');
                }
                tr.innerHTML = `
                    <td>${namesMap[key] || key} ${key === 'RandomForest' ? '⭐' : ''}</td>
                    <td>${(modelData.training_accuracy * 100).toFixed(1)}%</td>
                    <td>${(modelData.testing_accuracy * 100).toFixed(1)}%</td>
                    <td>${(modelData.precision * 100).toFixed(1)}%</td>
                    <td>${(modelData.recall * 100).toFixed(1)}%</td>
                    <td>${(modelData.f1_score * 100).toFixed(1)}%</td>
                `;
                tbody.appendChild(tr);
            });

            // Populate Feature Importances
            const importanceList = document.getElementById('importance-list');
            importanceList.innerHTML = ''; // Clear loading indicator

            // Find maximum importance to scale visual bars relative to it
            const maxVal = data.importances[0].importance;

            data.importances.forEach(item => {
                const row = document.createElement('div');
                row.className = 'importance-row';
                
                const percentageOfMax = (item.importance / maxVal) * 100;
                
                row.innerHTML = `
                    <div class="importance-name">${item.feature}</div>
                    <div class="importance-bar-wrapper">
                        <div class="importance-bar" style="width: 0%;" data-width="${percentageOfMax}%"></div>
                    </div>
                    <div class="importance-val">${(item.importance * 100).toFixed(1)}%</div>
                `;
                importanceList.appendChild(row);
            });

            // Trigger progress bar animations after rendering
            setTimeout(() => {
                const bars = document.querySelectorAll('.importance-bar');
                bars.forEach(bar => {
                    bar.style.width = bar.getAttribute('data-width');
                });
            }, 100);
        })
        .catch(err => {
            console.error("Error loading metrics:", err);
        });
}

// Load Sandbox Profiles
function loadSampleProfiles() {
    fetch('/api/samples?t=' + Date.now())
        .then(res => res.json())
        .then(samples => {
            const container = document.getElementById('samples-container');
            container.innerHTML = ''; // Clear loader text

            samples.forEach((sample, index) => {
                const btn = document.createElement('button');
                btn.className = 'sample-btn';
                btn.textContent = sample.name;
                btn.addEventListener('click', () => {
                    // Remove active from all sample buttons
                    document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    fillForm(sample);
                });
                container.appendChild(btn);
            });
        })
        .catch(err => {
            console.error("Error loading samples:", err);
        });
}

// Auto-fill form from selected sample profile
function fillForm(sample) {
    // Fill sliders
    const numericKeys = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion'];
    numericKeys.forEach(key => {
        const slider = document.getElementById(key);
        const valEl = document.getElementById(`${key.toLowerCase()}-val`);
        if (slider) {
            slider.value = sample[key];
            if (valEl) valEl.textContent = sample[key];
        }
    });

    // Fill dropdowns
    const dropdownKeys = [
        'gender', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 
        'Relation', 'StudentAbsenceDays', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction'
    ];
    dropdownKeys.forEach(key => {
        const dropdown = document.getElementById(key);
        if (dropdown) {
            dropdown.value = sample[key];
        }
    });
}

// Analyze Performance (Make API call)
function analyzeStudent() {
    const btn = document.getElementById('submit-btn');
    btn.disabled = true;
    btn.textContent = "Processing details...";

    // 1. Gather Inputs
    const payload = {
        raisedhands: parseInt(document.getElementById('raisedhands').value),
        VisITedResources: parseInt(document.getElementById('VisITedResources').value),
        AnnouncementsView: parseInt(document.getElementById('AnnouncementsView').value),
        Discussion: parseInt(document.getElementById('Discussion').value),
        gender: document.getElementById('gender').value,
        StageID: document.getElementById('StageID').value,
        GradeID: document.getElementById('GradeID').value,
        SectionID: document.getElementById('SectionID').value,
        Topic: document.getElementById('Topic').value,
        Semester: document.getElementById('Semester').value,
        Relation: document.getElementById('Relation').value,
        StudentAbsenceDays: document.getElementById('StudentAbsenceDays').value,
        ParentAnsweringSurvey: document.getElementById('ParentAnsweringSurvey').value,
        ParentschoolSatisfaction: document.getElementById('ParentschoolSatisfaction').value
    };

    // 2. HTTP POST Request
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        btn.disabled = false;
        btn.textContent = "Analyze Performance & advising";

        if (data.status === 'success') {
            displayResults(data);
        } else {
            alert("Prediction error: " + data.message);
        }
    })
    .catch(err => {
        btn.disabled = false;
        btn.textContent = "Analyze Performance & advising";
        console.error("API request failed:", err);
        alert("API connection failed. Is the Flask server running?");
    });
}

// Render dynamic results in UI
function displayResults(data) {
    // 1. Update Result Banner Class & Text
    const banner = document.getElementById('result-banner');
    banner.className = 'result-banner'; // Reset classes
    
    // Add specific class based on prediction index
    if (data.prediction === 0) {
        banner.classList.add('low-class');
    } else if (data.prediction === 1) {
        banner.classList.add('medium-class');
    } else {
        banner.classList.add('high-class');
    }
    
    document.getElementById('predicted-label').textContent = data.prediction_label;
    document.getElementById('confidence-score').textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // 2. Render BFS Stepper Progress
    const stepper = document.getElementById('bfs-stepper');
    stepper.innerHTML = ''; // Clear previous steps
    
    data.advising_path.forEach((step, index) => {
        // Render step node
        const node = document.createElement('div');
        node.className = 'step-node';
        node.textContent = step;
        
        // Highlight active or pass node
        if (index === 0) {
            node.classList.add('active-node');
        }
        if (step === 'Pass') {
            node.classList.add('goal-node');
        }
        stepper.appendChild(node);
        
        // Render connector arrow if not last node
        if (index < data.advising_path.length - 1) {
            const arrow = document.createElement('span');
            arrow.className = 'step-arrow';
            arrow.innerHTML = '➔';
            stepper.appendChild(arrow);
        }
    });

    // 3. Render CSP Planner Schedule
    document.getElementById('csp-hours').textContent = `${data.csp_plan.StudyHours} Hrs/Day`;
    document.getElementById('csp-tutoring').textContent = data.csp_plan.Tutoring;
    document.getElementById('csp-monitoring').textContent = data.csp_plan.Monitoring;
    
    const attMap = { "Low": "Normal Attendance", "Medium": "Target Med Attendance", "High": "Target High Attendance" };
    document.getElementById('csp-attendance').textContent = attMap[data.csp_plan.Attendance] || data.csp_plan.Attendance;

    // 4. Render Forward Chaining Inference Log
    const initialFactsList = document.getElementById('initial-facts-list');
    initialFactsList.innerHTML = ''; // Clear default None
    data.initial_facts.forEach(fact => {
        const span = document.createElement('span');
        span.className = 'fact-badge';
        span.textContent = fact;
        initialFactsList.appendChild(span);
    });

    const inferenceLog = document.getElementById('inference-log');
    inferenceLog.innerHTML = ''; // Clear default
    
    if (data.inference_trace.length === 0) {
        const li = document.createElement('li');
        li.className = 'placeholder-text';
        li.textContent = "No rules fired (default state matches initial facts).";
        inferenceLog.appendChild(li);
    } else {
        data.inference_trace.forEach(step => {
            const li = document.createElement('li');
            li.className = 'log-item';
            
            const conds = step.conditions.join(" & ");
            li.innerHTML = `Rule Fired: <span class="rule-str">{${conds}}</span> ➔ Derived Fact: <span class="concl-str">${step.conclusion}</span>`;
            inferenceLog.appendChild(li);
        });
    }

    // Scroll smoothly to output
    document.querySelector('.tab-navigation').scrollIntoView({ behavior: 'smooth' });
}
