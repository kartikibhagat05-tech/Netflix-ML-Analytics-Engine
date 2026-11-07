import streamlit as st
import pandas as pd
import numpy as np

# सभी कोर टास्क मॉड्यूल्स को इम्पोर्ट किया
from task1_recommendation import NetflixContentRecommender
from task2_type_prediction import NetflixTypePredictor
from task3_rating_classification import NetflixRatingClassifier
from task4_segmentation import NetflixContentSegmentor
from task5_forecasting import NetflixTrendForecaster
from task6_success_engine import NetflixSuccessAnalyticsEngine

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Netflix ML Analytics Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CORPORATE LOOK ---
import base64

# इमेजला Base64 मध्ये कन्वर्शन करणारा कोड
import base64
import os

import base64
import os
import glob

# इमेजला Base64 मध्ये कन्वर्शन करणारा कोड
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, "static")
    
    # static फोल्डरमध्ये 'background' नावाने सुरू होणारी कोणतीही फाईल शोधणे (एक्स्टेंशन काहीही असो)
    search_pattern = os.path.join(static_dir, "background.*")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"static फोल्डरमध्ये 'background' नावाची कोणतीही इमेज सापडली नाही.")
    
    # पहिली सापडलेली फाईल वापरणे
    full_image_path = matching_files[0]
    img_base64 = get_base64_image(full_image_path)
    
    # --- CUSTOM CSS WITH BASE64 BACKGROUND ---
    st.markdown(f"""
    <style>
    /* मुख्य ॲप बॅकग्राउंड */
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
    }}
    
    /* वरची पांढरी पट्टी (Header) पारदर्शक करणे */
    header[data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}
    
    /* डावीकडील नेव्हिगेशन पॅनेल (Sidebar) पारदर्शक करणे */
    section[data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.15) !important; /* फिकट काचेसारखा लुक */
        backdrop-filter: blur(10px); /* सिडेबारच्या मागे हलका ब्लर इफेक्ट */
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* सिडेबारमधील टेक्स्टचा रंग (वाचायला सोपा जाण्यासाठी) */
    section[data-testid="stSidebar"] .st-emotion-cache-17l7u4g, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {{
        color: #111111 !important;
    }}

    /* इतर बॉक्स आणि कंटेटचे स्टायलिंग */
    .main-title {{ font-size:38px !important; font-weight: 700; color: #E50914; margin-bottom: 5px; }}
    .subtitle {{ font-size:18px !important; color: #333; margin-bottom: 25px; font-weight: 500; }}
    .metric-box {{ background-color: rgba(248, 249, 250, 0.85); padding: 15px; border-radius: 10px; border-left: 5px solid #E50914; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    .insight-box {{ background-color: rgba(238, 247, 255, 0.85); padding: 15px; border-radius: 8px; border-left: 5px solid #007mb8; margin-top: 10px; }}
    </style>
""", unsafe_allow_html=True)

except Exception as e:
    st.error(f"बॅकग्राउंड इमेज लोड करताना एरर आला: {e}")



# --- SIDEBAR ORCHESTRATION ---
st.sidebar.markdown("<h2 style='color: #E50914;'>🎬 Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.write("Navigate through various machine learning pipeline layers.")

menu_option = st.sidebar.radio(
    "Select Analytics Layer:",
    [
        "🚀 Global Success Engine (Task 6)",
        "🎯 Content Recommendation (Task 1)",
        "🔮 Format Type Prediction (Task 2)",
        "📊 Audience Rating Classifier (Task 3)",
        "🧩 Content Segmentation (Task 4)",
        "📈 Trend Forecasting (Task 5)"
    ]
)
# --- SIDEBAR ORCHESTRATION ---
st.sidebar.markdown("<h2 style='color: #E50914;'>🎬 Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.write("Navigate through various machine learning pipeline layers.")

# नवीन फिचर: रॉ डेटा पाहण्यासाठी साईडबारमध्ये चेकबॉक्स दिला
show_raw_data = st.sidebar.checkbox("📂 View Netflix Raw Dataset")

# युझरने चेकबॉक्स टिक केल्यास मुख्य स्क्रीनवर डेटाफ्रेम दिसेल
@st.cache_resource
def get_analytics_engine():
    return NetflixSuccessAnalyticsEngine("data.csv")

try:
    master_engine = get_analytics_engine()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'data.csv' is in the project root.")
    st.stop()

if show_raw_data:
    st.markdown("<h3 style='color: white;'>📊 Dataset Explorer (Top 5 Records)</h3>", unsafe_allow_html=True)
    try:
        # master_engine च्या आत असलेल्या डेटाफ्रेममधून पहिली ५ रोज दाखवणे
        st.dataframe(master_engine.t2_engine.df.head(5), width='stretch')
        st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    except Exception as e:
        st.warning("Could not render preview data layer.")


# =====================================================================
# 🚀 LAYER 6: GLOBAL SUCCESS ENGINE (TASK 6)
# =====================================================================
if menu_option == "🚀 Global Success Engine (Task 6)":
    st.markdown("<div class='main-title'>Enterprise MLOps Orchestration Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>End-to-End Machine Learning Pipeline Execution & Resource Audit</div>", unsafe_allow_html=True)
    
    st.write("Click below to compile all sub-systems, analyze cross-functional model metrics, and generate real-time corporate insights.")
    
    if st.button("Execute End-to-End Pipeline", type="primary"):
        with st.spinner("Compiling cross-functional arrays & training predictive models..."):
            # Task 6 का मुख्य फंक्शन रन करें
            report = master_engine.execute_end_to_end_analytics_pipeline()
            
            st.success("✅ End-to-End Pipeline compiled successfully inside MLOps environments.")
            
            # KPI Metrics Cards Layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='metric-box'><h5>⏱️ Training Latency</h5><h3>{:.2f} ms</h3></div>".format(report["Total Pipeline Compile Time (ms)"]), unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-box'><h5>📦 Aggregated RAM Usage</h5><h3>{:.2f} KB</h3></div>".format(report["Aggregated RAM Allocation Footprint (KB)"]), unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-box'><h5>📈 Trend Fit (R²)</h5><h3>{:.4f}</h3></div>".format(report["Task 5 Nonlinear Trend R² Score"]), unsafe_allow_html=True)
            
            # Model Accuracy Comparison Table
            st.subheader("📊 Accuracy & Performance Evaluation Matrix")
            metrics_data = {
                "Model / Task Layer": [
                    "Task 2: Random Forest (Format Match)", 
                    "Task 2: Logistic Regression (Base Match)", 
                    "Task 3: Tuned Random Forest (Rating Match)"
                ],
                "Accuracy Score (%)": [
                    f"{report['Task 2 Random Forest Format Accuracy (%)']:.2f} %",
                    f"{report['Task 2 Logistic Regression Base Accuracy (%)']:.2f} %",
                    f"{report['Task 3 Tuned Multi-Class Rating Accuracy (%)']:.2f} %"
                ]
            }
            st.table(pd.DataFrame(metrics_data))
            
            # Automated Intelligent Business Insights Generation
            st.subheader("💡 Automated Intelligent Business Insights")
            for insight in report["Automated Corporate Insights Log"]:
                st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

# =====================================================================
# 🎯 LAYER 1: CONTENT RECOMMENDATION (TASK 1)
# =====================================================================
elif menu_option == "🎯 Content Recommendation (Task 1)":
    st.markdown("<div class='main-title'>NLP Content Recommendation Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>TF-IDF Vector Space Multi-threading Cosine Similarity Search</div>", unsafe_allow_html=True)
    
    # NLP Matrix Build Trigger
    build_ms, space_kb = master_engine.t1_engine.execute_nlp_vectorization()
    st.caption(f"⚙️ NLP Model Fit Matrix: Generated in {build_ms:.2f} ms | RAM Footprint: {space_kb:.2f} KB")
    
    all_titles = master_engine.t1_engine.df['title'].dropna().unique()
    target_movie = st.selectbox("Type or select a movie/show title:", all_titles)
    top_k = st.slider("Number of recommendations:", 3, 10, 5)
    
    if st.button("Generate Similarity Matrix"):
        # spinner जोडला
        with st.spinner("Calculating cosine similarity matrix vectors..."):
            recs, runtime, ram = master_engine.t1_engine.get_similarity_recommendations(target_movie, top_n=top_k)
            if recs is not None:
                st.write(f"🏎️ **Latency:** {runtime:.2f} ms | **Live RAM:** {ram:.2f} KB")
                st.dataframe(recs, width='stretch')
            else:
                st.warning("Title not found or couldn't compute similarity.")


# =====================================================================
# 🔮 LAYER 2: FORMAT TYPE PREDICTION (TASK 2)
# =====================================================================
elif menu_option == "🔮 Format Type Prediction (Task 2)":
    st.markdown("<div class='main-title'>Format Type Prediction Layer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict whether content is a Movie or TV Show using Random Forest</div>", unsafe_allow_html=True)
    
    # Train if not trained
    lr_acc, _, rf_acc, _, _, _ = master_engine.t2_engine.train_and_compare_models()
    
    col1, col2 = st.columns(2)
    with col1:
        input_rating = st.selectbox("Select Target Rating:", master_engine.t2_engine.df['rating'].unique())
        input_year = st.number_input("Select Release Year:", min_value=1900, max_value=2030, value=2021)
    with col2:
        input_country = st.selectbox("Select Target Country:", master_engine.t2_engine.df['country'].unique())

    if st.button("Predict Format Class"):
        with st.spinner("Running Random Forest inference matrix..."):
            pred, latency = master_engine.t2_engine.predict_format(input_rating, input_year, input_country)
            st.info(f"🔮 **Predicted Content Format:** **{pred}**")
            st.caption(f"⚡ Inference Latency: {latency:.4f} ms | Trained RF Accuracy: {rf_acc:.2f}%")


# =====================================================================
# 📊 LAYER 3: AUDIENCE RATING CLASSIFIER (TASK 3)
# =====================================================================
elif menu_option == "📊 Audience Rating Classifier (Task 3)":
    st.markdown("<div class='main-title'>Audience Rating Classification Layer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Multi-Class Hyperparameter Tuned Parental Target Audience Predictor</div>", unsafe_allow_html=True)
    
    acc, _, _, _ = master_engine.t3_engine.train_optimized_classification_model()
    
    col1, col2 = st.columns(2)
    with col1:
        in_type = st.selectbox("Content Type:", master_engine.t3_engine.df['type'].unique())
        in_year = st.number_input("Release Year:", min_value=1900, max_value=2030, value=2021, key="t3_year")
    with col2:
        in_country = st.selectbox("Country Origin:", master_engine.t3_engine.df['country'].unique(), key="t3_country")

        if st.button("Predict Audience Rating"):
            with st.spinner("Processing multi-class hyperparameter boundaries..."):
                pred_rating, latency = master_engine.t3_engine.predict_audience_rating(in_type, in_year, in_country)
                if isinstance(pred_rating, str) and "Error" in pred_rating:
                    st.error(pred_rating)
                else:
                    st.success(f"🎯 **Predicted Target Parental Category:** **{pred_rating}**")
                    st.caption(f"⚡ Model Training Base Accuracy: {acc:.2f}% | Process Latency: {latency:.4f} ms")



# =====================================================================
# 🧩 LAYER 4: CONTENT SEGMENTATION (TASK 4)
# =====================================================================
elif menu_option == "🧩 Content Segmentation (Task 4)":
    st.markdown("<div class='main-title'>Unsupervised Content Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>K-Means Feature Cluster Framework Normalization Analysis</div>", unsafe_allow_html=True)
    
    clusters_count = st.slider("Select K-Means Cluster Size (Clusters):", 2, 6, 4)
    wcss, duration, ram = master_engine.t4_engine.build_unsupervised_segmentation(n_clusters=clusters_count)
    
    st.write(f"📈 **WCSS Inertia Score:** {wcss:.4f} | **Latency:** {duration:.2f} ms | **Model RAM:** {ram:.2f} KB")
    
    selected_cluster = st.selectbox("Inspect Cluster Segment Characteristics:", range(clusters_count))
    total, preview_table = master_engine.t4_engine.interpret_segment_characteristics(selected_cluster)
    
# =====================================================================
# 📈 LAYER 5: TREND FORECASTING (TASK 5)
# =====================================================================
# =====================================================================
# 📈 LAYER 5: TREND FORECASTING (TASK 5)
# =====================================================================
elif menu_option == "📈 Trend Forecasting (Task 5)":
    st.markdown("<div class='main-title'>Nonlinear Trend Forecasting Layer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predicting Netflix Content Growth and Release Volumes using Machine Learning</div>", unsafe_allow_html=True)
    
    st.write("Select a target year to forecast the future volume of content releases based on historical trends.")
    
    # युझरला वर्ष निवडण्यासाठी स्लायडर दिला (२०२४ ते २०३५)
    target_year = st.slider("Select Target Year for Forecast:", min_value=2024, max_value=2035, value=2027)
    
    if st.button("Generate Trend Forecast", type="primary"):
        with st.spinner("Analyzing time-series layers & calculating nonlinear projections..."):
            # Task 5 च्या इंजिनमधून प्रेडिक्शन आणि लॅटन्सी आणली
            projected_count, latency = master_engine.t5_engine.forecast_future_trends(target_year)
            
            # सुरक्षितता: जर प्रेडिक्शन एरर स्ट्रिंग असेल तर एरर बॉक्स दाखवा, आकडा असेल तर रिझल्ट कार्ड्स दाखवा
            if isinstance(projected_count, str):
                st.error(projected_count)
            else:
                # रिझल्ट दाखवण्यासाठी अ‍ॅस्थेटिक कार्ड्स
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(
                        f"<div class='metric-box'><h5>🔮 Projected Release Volume</h5><h3>{int(projected_count)} Titles</h3></div>", 
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"<div class='metric-box'><h5>⚡ Inference Latency</h5><h3>{latency:.4f} ms</h3></div>", 
                        unsafe_allow_html=True
                    )
                
                # एक सुंदर कॉर्पोरेट इनसाईट बॉक्स
                st.markdown(
                    f"<div class='insight-box'>💡 <b>Trend Analysis Insight:</b> Based on historical data pattern recognition, the estimated production pipeline capacity for the year {target_year} safely tracks at approximately {int(projected_count)} new catalog expansions.</div>", 
                    unsafe_allow_html=True
                )
