## Iris Flowers Classification Project
# 📌 Project Overview
This project focuses on classifying Iris flower species using the classic Iris dataset (150 samples, 4 features, 3 species). The project demonstrates a complete machine learning workflow—from data preprocessing and visualization to model training using Support Vector Machines (SVM) with pipelines.

The solution is efficient, reproducible, and scalable, making it suitable for portfolio demonstration or freelance work.


#🛠️ Key Features
Data Preprocessing: Cleaned data and encoded target labels.

Pipeline Integration: Streamlined preprocessing and model training using Scikit-learn pipelines.

Exploratory Data Analysis (EDA): Visualizations including pairplots, histograms, boxplots, and correlation heatmaps.

Modeling: SVM classifier to predict Iris species with high accuracy.

Evaluation: Achieved R² score of 1.0% (classification accuracy near 100%).


# 📊 Visual Insights
Feature distributions and class separations.

Relationships between sepal and petal dimensions.

Interactive plots for clear data understanding.


# 🧩 Workflow

Load Data: Iris dataset (150 samples, 4 features).

Preprocess: Encode target, scale features, setup pipelines.

EDA & Visualization: Analyze feature distributions and correlations.

Model Training: Train SVM classifier with pipeline.

Evaluation: Accuracy and R² score assessment.

Prediction: Predict new Iris samples efficiently.


## 📂 Project Structure
Iris-Classification/
│
├─ data/
│   └─ iris.csv
├─ notebooks/
│   └─ iris_eda.ipynb
├─ models/
│   └─ svm_pipeline.pkl
├─ plots/
│   └─ feature_analysis.png
├─ app.py  (optional Streamlit app)
└─ README.md

#⚙️ How to Run
1. Clone the repository:
git clone [https://github.com/daanni69/iris-flower-classification-model/tree/main](https://github.com/daanni69/iris-flower-classification-model/tree/main)

2. Install dependencies:
pip install -r requirements.txt

3. Run the notebook or script:
python iris_pipeline.py









