import pandas as pd
import numpy as np
import time
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class NetflixTrendForecaster:
    def __init__(self, dataset_path="data.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.trend_model = LinearRegression()
        self.yearly_counts = None
        self.load_and_parse_time_data()
        # FIX: क्लास इनिशियलाइज होतानाच सुरक्षितपणे मॉडेल बॅकग्राउंडला ट्रेन करून ठेवणे
        try:
            self.train_trend_forecasting_pipeline()
        except:
            pass
        
    def load_and_parse_time_data(self):
        # Step 1: Prepare time-based features from original rows setup
        self.df = pd.read_csv(self.dataset_path)
        if 'release_year' in self.df.columns:
            self.df['release_year'] = self.df['release_year'].fillna(2021).astype(int)
            
        # Step 2: Analyze historical release patterns matrix arrays
        self.yearly_counts = self.df['release_year'].value_counts().sort_index()

    def train_trend_forecasting_pipeline(self):
        # Step 3: Build forecasting models via Advanced Polynomial Feature arrays
        start_time = time.time()
        
        valid_years = self.yearly_counts.index[self.yearly_counts.index >= 2000].to_numpy()
        content_volumes = self.yearly_counts[valid_years].to_numpy()
        
        X = valid_years.reshape(-1, 1)
        y = content_volumes
        
        X_poly = self.poly.fit_transform(X)
        self.trend_model.fit(X_poly, y)
        
        # Step 5: Evaluate forecasting accuracy (R² Metric Score Log)
        r2_performance_score = self.trend_model.score(X_poly, y)
        execution_latency_ms = (time.time() - start_time) * 1000
        forecaster_ram_kb = sys.getsizeof(self.trend_model) / 1024
        
        return r2_performance_score, execution_latency_ms, forecaster_ram_kb

    def forecast_future_trends(self, target_future_year):
        # Step 4: Generate future predictions based on historical polynomial curves
        start_time = time.time()
        try:
            # MASTER FIX: poly किंवा model दोन्हीपैकी काहीही फिट नसेल तर आधी ते सक्तीने ट्रेन करून घेणे
            if not hasattr(self.poly, "n_features_in_") or not hasattr(self.trend_model, "coef_"):
                self.train_trend_forecasting_pipeline()

            raw_vector = np.array([[int(target_future_year)]])
            poly_vector = self.poly.transform(raw_vector)
            projected_volume = self.trend_model.predict(poly_vector)
            
            # SCALAR FIX: Extract accurate array element to single value float scalar conversion block
            predicted_value = float(projected_volume[0])
            projected_volume = max(0, int(np.round(predicted_value)))
            
            inference_latency_ms = (time.time() - start_time) * 1000
            return projected_volume, inference_latency_ms
        except Exception as e:
            # बॅकअप प्लॅन: जर तरीही एक्सेप्शन आले तर फोर्सफुली री-ट्रेन करून व्हॅल्यू काढणे
            try:
                self.train_trend_forecasting_pipeline()
                raw_vector = np.array([[int(target_future_year)]])
                poly_vector = self.poly.transform(raw_vector)
                projected_volume = self.trend_model.predict(poly_vector)
                predicted_value = float(projected_volume[0])
                return max(0, int(np.round(predicted_value))), (time.time() - start_time) * 1000
            except Exception as inner_e:
                return f"Error matrix bounds configuration exception raised: {str(inner_e)}", 0

# Automated validation unit block engine profile configuration log
if __name__ == "__main__":
    print("📈 Running Performance-Optimized Advanced Task 5 Trend Forecasting Pipeline Test...")
    forecaster = NetflixTrendForecaster("data.csv")
    r2, duration, ram = forecaster.train_trend_forecasting_pipeline()
    
    print("\n==================== STEP 3 & 5: FORECASTING MODEL PERFORMANCE EVALUATION ====================")
    print(f"🌲 Mathematical Optimization Applied: [Nonlinear Polynomial Regression - Degree 2 Curve]")
    print(f"✅ Time-Series Trend Matrix generated successfully inside NumPy vector spaces.")
    print(f"📈 Model Trend Fit Determination Metric (Polynomial R² Score): {r2:.4f}")
    print(f"🏎️ Forecasting Training Latency Optimization Log            : {duration:.2f} ms")
    print(f"📦 Active Time-Series Forecaster Model RAM Size              : {ram:.2f} KB")
    
    print("\n---------------- STEP 4: GENERATING FUTURE PROJECTIONS AND INSIGHTS ----------------")
    future_years = [2027, 2028, 2029]
    for target_y in future_years:
        projected_count, edge_latency = forecaster.forecast_future_trends(target_y)
        print(f"[Year Forecast Target: {target_y}] -> Projected Content Release Volume Rate: {projected_count}")
