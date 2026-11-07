# 🎬 Netflix Enterprise ML Analytics Engine

An advanced, production-grade end-to-end Machine Learning pipeline and telemetry application built over the comprehensive global Netflix metadata repository (`data.csv`). The core framework integrates four distinct machine learning domains across a six-tier decoupled backend module layer, managed via a high-performance **Glassmorphism Tech-Aesthetic UI dashboard**.

## 🚀 Key Pipeline Task Modules
1. **Global Success Engine (Task 6):** Coordinates automated pipeline compilation, monitoring localized inference latency parameters alongside total multi-model RAM footprint audits.
2. **Content Recommendation (Task 1):** Executes text mining over metadata pooling arrays utilizing TF-IDF token processing and Cosine Similarity computations.
3. **Format Type Prediction (Task 2):** Evaluates format bounds using structural features passed to trained Random Forest and Logistic Regression classifiers.
4. **Audience Rating Classifier (Task 3):** Handles multi-class parental category indexing through optimized hyperparameter constraints (`max_depth=12`, `criterion='entropy'`).
5. **Content Segmentation (Task 4):** Groups content clusters programmatically via unsupervised K-Means algorithms layered with data scaling standard transformations.
6. **Trend Forecasting (Task 5):** Maps historic title growth parameters over custom 2nd-degree Polynomial Feature curvilinear regressions to project multi-year distribution capacities.

## 🛠️ Step-by-Step System Installation Guide

### 1. Clone Development Directory
```bash
git clone https://github.com/your-username/Netflix-ML-Analytics-Engine.git
cd Netflix-ML-Analytics-Engine
```

### 2. Configure Virtual Environment & Packages
Ensure `data.csv` is correctly positioned within the root directory before running environment setups.
```bash
# Initialize Python isolated virtual environment
python -m venv venv

# Activate execution shell track (Windows)
venv\Scripts\activate

# Activate execution shell track (Linux / macOS)
source venv/bin/activate

# Upgrade distribution setup and fetch locked requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch UI Engine
```bash
streamlit run app.py
```
Open your browser endpoint at `http://localhost:8501` to interface with the active pipeline modules.
