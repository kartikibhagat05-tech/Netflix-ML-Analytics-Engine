import pandas as pd
import numpy as np
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NetflixContentRecommender:
    def __init__(self, dataset_path="data.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.load_and_preprocess()
        
    def load_and_preprocess(self):
        # Company cha original dataset read karne
        self.df = pd.read_csv(self.dataset_path)
        
        # Preprocessing text array fields safely
        for col in ['listed_in', 'director', 'country']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')
        self.df['title'] = self.df['title'].fillna('Untitled')
        
        # NLP Feature pool aggregation text corpus string generation
        self.df['combined_features'] = (
            self.df['listed_in'] + " " + 
            self.df['director'] + " " + 
            self.df['country']
        ).str.lower()

    def execute_nlp_vectorization(self):
        # Space-Time Analytics profiling block tracking log
        start_time = time.time()
        
        # TF-IDF calculation vector array embedding pipeline setup
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        
        build_time = (time.time() - start_time) * 1000  # ms convert
        matrix_ram_size = sys.getsizeof(self.tfidf_matrix) / 1024  # KB conversion
        return build_time, matrix_ram_size

    def get_similarity_recommendations(self, target_title, top_n=5):
        start_time = time.time()
        title_lower = target_title.lower()
        
        matched_indices = self.df[self.df['title'].str.lower() == title_lower].index
        if len(matched_indices) == 0:
            return None, 0, 0
            
        idx = matched_indices
        # Real-time Cosine Similarity Vector Multi-threading execution logic computation
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Argson reverse sorting data frames generation extraction blocks
        similar_indices = cosine_sim.argsort()[:-(top_n+2):-1]
        similar_indices = [i for i in similar_indices if i != idx][:top_n]
        
        results = self.df.iloc[similar_indices][['title', 'type', 'listed_in', 'rating', 'release_year']].copy()
        results['Similarity Score'] = np.round(cosine_sim[similar_indices], 2)
        
        execution_query_time = (time.time() - start_time) * 1000  # ms Latency Log
        memory_footprint_used = (sys.getsizeof(cosine_sim) + sys.getsizeof(results)) / 1024  # KB Space Log
        
        return results, execution_query_time, memory_footprint_used

# Standalone execution unit logic for company tracking benchmark proofing verification logs
if __name__ == "__main__":
    print("🎬 Running Standalone Task 1 Performance Test...")
    engine = NetflixContentRecommender("data.csv")
    build_ms, space_kb = engine.execute_nlp_vectorization()
    print(f"✅ NLP Model Fit Matrix generated in: {build_ms:.2f} ms | RAM footprints allocated: {space_kb:.2f} KB")
    
    # Dataset मधल्या पहिल्या मुव्ही वर टेस्ट करू (उदा. 'Dick Johnson Is Dead')
    first_title = engine.df['title'].iloc[0]
    recs, runtime, ram = engine.get_similarity_recommendations(first_title)
    if recs is not None:
        print(f"\n🎯 Top Recommendations for '{first_title}':")
        print(recs.to_string(index=False))
        print(f"\n🏎️ Query Execution Latency Rate: {runtime:.2f} ms | Live Ram Space: {ram:.2f} KB")
