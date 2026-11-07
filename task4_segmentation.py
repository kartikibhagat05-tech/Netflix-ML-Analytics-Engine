import pandas as pd
import numpy as np
import time
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetflixContentSegmentor:
    def __init__(self, dataset_path="data.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        # Step 1: Prepare numerical and categorical features framework array pipeline
        self.df = pd.read_csv(self.dataset_path)
        
        # Missing entries clean up matrix data alignment
        for col in ['type', 'country', 'rating', 'listed_in']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        self.df['release_year'] = self.df['release_year'].fillna(2021)

    def build_unsupervised_segmentation(self, n_clusters=4):
        # Step 2: Apply clustering algorithms with scaled feature array spaces
        start_time = time.time()
        
        features_df = self.df[['release_year', 'country', 'type']].copy()
        
        # Matrix vector space calculations sathi encoder arrays register mappings
        for col in ['country', 'type']:
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col].astype(str))
            
        # Data scaling normalization execution trace checks block
        scaled_features = self.scaler.fit_transform(features_df.astype(float).to_numpy())
        
        # Step 3: Identify content groups via K-Means algorithm framework configurations
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        self.df['cluster_id'] = self.kmeans_model.fit_predict(scaled_features)
        
        # Step 4: Visualize/Evaluate cluster performance matrix parameters logs
        wcss_score = self.kmeans_model.inertia_  # Mathematical baseline score footprint metrics
        
        execution_latency_ms = (time.time() - start_time) * 1000
        cluster_ram_size_kb = sys.getsizeof(self.kmeans_model) / 1024
        
        return wcss_score, execution_latency_ms, cluster_ram_size_kb

    def interpret_segment_characteristics(self, segment_id):
        # Step 5: Interpret cluster characteristics inside target data array splits
        cluster_subset = self.df[self.df['cluster_id'] == segment_id]
        total_items = len(cluster_subset)
        
        # Extract characteristic patterns logs metrics profiles summary
        sample_preview = cluster_subset[['title', 'type', 'release_year', 'country', 'rating']].head(3)
        return total_items, sample_preview

if __name__ == "__main__":
    print("🧩 Running Performance-Optimized Task 4 Unsupervised Content Segmentation Script...")
    segmentor = NetflixContentSegmentor("data.csv")
    
    # Standalone execution benchmarking layout trace configuration maps verify
    target_cluster_count = 4
    wcss, duration, ram = segmentor.build_unsupervised_segmentation(n_clusters=target_cluster_count)
    
    print("\n==================== STEP 4: UNSUPERVISED METRIC INTERPRETATION AND VERIFICATION ====================")
    print(f"✅ Unsupervised K-Means Cluster Matrix generated successfully inside NumPy matrix frames.")
    print(f"📈 Within-Cluster Sum of Squares (WCSS Inertia Score): {wcss:.4f}")
    print(f"🏎️ Matrix Optimization Pipeline Latency Latency Log: {duration:.2f} ms")
    print(f"📦 Active Unsupervised Model RAM Footprint size   : {ram:.2f} KB")
    
    print("\n==================== STEP 5: INTERPRET CLUSTER CHARACTERISTICS DATA SHOWCASE ====================")
    for idx in range(target_cluster_count):
        total, preview_table = segmentor.interpret_segment_characteristics(idx)
        print(f"\n📂 [Cluster Segment Block ID: {idx}] -> Total Content Items Partitioned: {total}")
        print(preview_table.to_string(index=False))
