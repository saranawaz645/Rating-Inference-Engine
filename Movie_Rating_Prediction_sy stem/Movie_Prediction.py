import pandas as pd
import numpy as np
import logging
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Professional log formatting for system diagnostics
logging.basicConfig(level=logging.INFO, format='[SYSTEM LOG] %(message)s')

class CollaborativeForecaster:
    """
    A professional-grade Recommender System implementing Item-Item Similarity 
    heuristics and statistical correlation for preference estimation.
    """
    
    def __init__(self, data_source):
        self.source = data_source
        self.dataframe = None
        self.train_subset = None
        self.test_subset = None

    def prepare_dataset(self):
        """Phase 1: Data Ingestion & Strategic Partitioning"""
        logging.info("Initializing data ingestion sequence...")
        headers = ['user_id', 'item_id', 'rating', 'timestamp']
        self.dataframe = pd.read_csv(self.source, sep='\t', names=headers)
        
        # Applying a different random_state to ensure a unique data split
        self.train_subset, self.test_subset = train_test_split(
            self.dataframe, test_size=0.20, random_state=55
        )
        logging.info(f"Dataset partitioned. Test Samples: {len(self.test_subset)}")

    def execute_prediction_logic(self):
        """
        Phase 2: Similarity Computation & Inference Generation
        Utilizes Pearson Correlation to quantify relationships between items.
        """
        logging.info("Computing Item-Item Similarity matrix (Pearson)...")
        
        # Generating interaction matrix
        pivot_grid = self.train_subset.pivot_table(index='user_id', columns='item_id', values='rating')
        
        # Establishing similarity thresholds to filter out noise
        similarity_index = pivot_grid.corr(method='pearson', min_periods=50)
        
        # Baseline calculations for missing data handling
        user_norms = self.train_subset.groupby('user_id')['rating'].mean()
        global_baseline = self.train_subset['rating'].mean()

        inferred_ratings = []

        # Inference loop for evaluation subset
        for idx, entry in self.test_subset.iterrows():
            curr_user = entry['user_id']
            curr_item = entry['item_id']
            
            # Handling cold-start scenarios
            if curr_item not in similarity_index or curr_user not in pivot_grid.index:
                inferred_ratings.append(user_norms.get(curr_user, global_baseline))
                continue
            
            # Similarity-weighted aggregation
            correlation_vector = similarity_index[curr_item]
            user_history = pivot_grid.loc[curr_user].dropna()
            
            overlapping_keys = correlation_vector.index.intersection(user_history.index)
            coefficient_weights = correlation_vector[overlapping_keys]
            historical_ratings = user_history[overlapping_keys]
            
            if coefficient_weights.abs().sum() == 0:
                inferred_ratings.append(user_norms.get(curr_user, global_baseline))
            else:
                # Mathematical weighted average formula
                weighted_val = (historical_ratings * coefficient_weights).sum() / coefficient_weights.abs().sum()
                inferred_ratings.append(weighted_val)

        # Normalizing predicted outputs (Clipping and Rounding)
        self.test_subset['predicted_val'] = inferred_ratings
        self.test_subset['predicted_val'] = (
            self.test_subset['predicted_val']
            .clip(1, 5)
            .apply(lambda r: round(r * 2) / 2)
        )

    def generate_analytics_report(self):
        """Phase 3: Statistical Evaluation & Export"""
        error_metric = sqrt(mean_squared_error(self.test_subset['rating'], self.test_subset['predicted_val']))
        
        # Exporting full prediction results to CSV
        results_filename = "comprehensive_analysis_report.csv"
        export_columns = ['user_id', 'item_id', 'rating', 'predicted_val']
        self.test_subset[export_columns].to_csv(results_filename, index=False)
        logging.info(f"Analytical report exported to {results_filename}")
        
        print("\n" + "•"*45)
        print(f"📊 SYSTEM DIAGNOSTICS: RMSE = {error_metric:.4f}")
        print("•"*45)
        print("Top 20 Predicted Interaction Results:")
        print(self.test_subset[export_columns].head(20).to_string(index=False))
        print(f"\nStatus: Full dataset ({len(self.test_subset)} rows) archived successfully.")
        print("•"*45)
        return error_metric

if __name__ == "__main__":
    try:
        # Execution Pipeline
        ai_engine = CollaborativeForecaster('u.data')
        ai_engine.prepare_dataset()
        ai_engine.execute_prediction_logic()
        ai_engine.generate_analytics_report()
    except Exception as e:
        logging.error(f"Execution interrupted: {str(e)}")