import pandas as pd
import numpy as np
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class NetflixTypePredictor:
    def __init__(self, dataset_path="data.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.lr_model = None
        self.rf_model = None
        self.label_encoders = {}
        self.load_and_clean_data()
        
    def load_and_clean_data(self):
        # Company cha original dataset read karne
        self.df = pd.read_csv(self.dataset_path)
        
        # Step 1: Select relevant dataset features & handle missing strings array profiles
        for col in ['rating', 'country', 'type']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        self.df['release_year'] = self.df['release_year'].fillna(2021)

    def train_and_compare_models(self):
        # Step 2: Encode categorical variables using Label Encoding mechanism arrays
        X_df = self.df[['rating', 'release_year', 'country']].copy()
        
        # Target variable matrix strong array converted to clear string numpy format
        y = self.df['type'].astype(str).to_numpy()  
        
        for col in ['rating', 'country']:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            self.label_encoders[col] = le
            
        # CRITICAL FIX: Convert completely to standard integer numpy matrix to avoid PyArrow error
        X = X_df.astype(int).to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ==================== MODEL 1: LOGISTIC REGRESSION ====================
        lr_start = time.time()
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_model.fit(X_train, y_train)
        y_pred_lr = self.lr_model.predict(X_test)
        lr_acc = accuracy_score(y_test, y_pred_lr) * 100
        lr_latency = (time.time() - lr_start) * 1000
        
        # ==================== MODEL 2: RANDOM FOREST ====================
        rf_start = time.time()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf) * 100
        rf_latency = (time.time() - rf_start) * 1000
        
        # Step 4: Evaluate prediction performance parameters metrics report logs
        perf_report = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
        rf_ram = sys.getsizeof(self.rf_model) / 1024
        
        return lr_acc, lr_latency, rf_acc, rf_latency, rf_ram, perf_report

    def predict_format(self, input_rating, input_year, input_country):
        start_time = time.time()
        try:
            enc_country = self.label_encoders['country'].transform([input_country])[0] if input_country in self.label_encoders['country'].classes_ else 0
            enc_rating = self.label_encoders['rating'].transform([input_rating])[0] if input_rating in self.label_encoders['rating'].classes_ else 0
            
            inference_vector = np.array([[int(enc_rating), int(input_year), int(enc_country)]])
            
            predicted_class = self.rf_model.predict(inference_vector)
            inference_latency = (time.time() - start_time) * 1000
            
            return predicted_class[0], inference_latency
        except Exception as e:
            return f"Error matrix bounds configuration exception raised: {str(e)}", 0

# Automated validation unit block engine pipeline layout config test
if __name__ == "__main__":
    print("🤖 Running Comparative Production-Grade Task 2 Classification Pipeline Test...")
    predictor = NetflixTypePredictor("data.csv")
    lr_acc, lr_time, rf_acc, rf_time, rf_ram, report = predictor.train_and_compare_models()
    
    print("\n==================== STEP 3 & 4: MODEL ACCURACY PERFORMANCE EVALUATION ====================")
    print(f"📈 Logistic Regression Base Accuracy : {lr_acc:.2f} % | Computational Latency Log: {lr_time:.2f} ms")
    print(f"🌲 Random Forest Ensemble Accuracy  : {rf_acc:.2f} % | Computational Latency Log: {rf_time:.2f} ms")
    print(f"📦 Random Forest Active RAM Allocation footprint size: {rf_ram:.2f} KB")
    
    print("\n==================== STEP 5: ACCURACY MODEL COMPARISON BENCHMARK TABLE ====================")
    best_model = "Random Forest Classifier" if rf_acc >= lr_acc else "Logistic Regression"
    print(f"🏆 Comparative Winner Champion Selection Engine Logic: [{best_model}]")
    print(f"🎯 Margin Delta Variance Gain Rate: {abs(rf_acc - lr_acc):.2f}% change accuracy vector rate shifts.")
    
    # Live single testing execution validation lines tracking data trace checks check log parameter
    test_c = str(predictor.df['country'].iloc[0])
    test_r = str(predictor.df['rating'].iloc[0])
    test_y = int(predictor.df['release_year'].iloc[0])
    
    format_pred, edge_latency = predictor.predict_format(test_r, test_y, test_c)
    print(f"\n🔮 Dynamic Sample Inference Verification Logs -> Input: [Country: {test_c}, Rating: {test_r}, Year: {test_y}]")
    print(f"🎬 Output Class Predicted Format: '{format_pred}' | Real-time Latency Speed: {edge_latency:.4f} ms")
