🎬 Rating Inference Engine
Predictive Modeling via Item-Item Collaborative Filtering
This project implements a statistical engine designed to forecast user ratings by calculating correlations between historical item interactions. It leverages high-dimensional matrix operations to predict user preferences.

🔍 Mathematical Workflow
Interaction Matrix: Constructed a Sparse User-Item Matrix from the MovieLens 100k dataset to represent user-item preferences.

Statistical Logic: Applied Pearson Correlation Coefficient to quantify the similarity between different items in the catalog.

Predictive Heuristics: Developed a Similarity-Weighted Averaging logic that prioritizes ratings from highly correlated items for more accurate inference.

Validation Strategy: Implemented a rigorous 80/20 Train-Test Split to evaluate the model's generalizability on unseen data.

📉 Evaluation & Diagnostics
Dataset: 100,000 rating records.

Metric: Root Mean Squared Error (RMSE).

Benchmark Achievement: Achieved a validated RMSE of 1.17, aligning with standard collaborative filtering baselines.

Reporting: Automated generation of a full-scale inference report (comprehensive_analysis_report.csv) covering 20,000+ predictions.

⚙️ Technical Environment
Primary Stack: Python, scikit-learn, pandas

Methodology: Statistical Correlation, Matrix Manipulation, and Predictive Analytics.

👩‍💻 Developed By:
Sara Nawaz, BS Computational Mathematics | University of KarachiMachine Learning Enthusiast
