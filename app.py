# Iris Classification Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Page Config
st.set_page_config(page_title="Iris Classifier", layout="wide")

st.title("🌸 Iris Flower Classification App")
st.markdown("Professional dashboard showing **data exploration, model training, and predictions** for the Iris dataset.")

# Load Data
iris_data = load_iris(as_frame=True)
df = iris_data.frame
df["species"] = df["target"].map(dict(enumerate(iris_data.target_names)))
feature_cols = iris_data.feature_names

st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# Show Flower Images
st.subheader("🌺 Iris Flower Types")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Irissetosa1.jpg", caption="Iris-setosa", use_container_width=True)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris-versicolor", use_container_width=True)
with col3:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris-virginica", use_container_width=True)

# Plotly Visualizations
st.subheader("📈 Data Visualizations")

# Histogram
fig_hist = px.histogram(df, x="sepal length (cm)", color="species", marginal="box", nbins=20,
                        title="Distribution of Sepal Length by Species")
st.plotly_chart(fig_hist, use_container_width=True)

# Scatter plot
fig_scatter = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)", color="species",
                        size="petal length (cm)", hover_data=["petal width (cm)"],
                        title="Sepal Length vs Sepal Width with Petal Size")
st.plotly_chart(fig_scatter, use_container_width=True)

# Model Training
st.subheader("🤖 Model Training and Evaluation")

X = df[feature_cols]
y = df["species"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM (linear kernel)
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump({"model": model, "label_encoder": label_encoder, "feature_cols": feature_cols}, "iris_best_pipeline.joblib")

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = list(label_encoder.classes_)

fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=labels,
    y=labels,
    annotation_text=cm.astype(str),
    colorscale="Blues"
)
fig_cm.update_layout(title="Confusion Matrix", xaxis=dict(title="Predicted"), yaxis=dict(title="True"))
st.plotly_chart(fig_cm, use_container_width=True)

# Classification Report
st.text("📋 Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=labels))

# Feature Importance
st.subheader("📊 Feature Importance (SVM Linear Kernel)")

coef = np.abs(model.coef_).mean(axis=0)
importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": coef}).sort_values(by="Importance", ascending=False)

fig_importance = px.bar(importance_df, x="Feature", y="Importance", text="Importance", 
                        title="Feature Importance", color="Importance", color_continuous_scale="Blues")
st.plotly_chart(fig_importance, use_container_width=True)

# Decision Boundaries (2D)
st.subheader("🔹 Decision Boundary (Top 2 Features)")

top_features = importance_df['Feature'].head(2).tolist()
X_train_2 = X_train[top_features]
X_test_2 = X_test[top_features]

model_2 = SVC(kernel="linear", probability=True, random_state=42)
model_2.fit(X_train_2, y_train)

# Create grid
x_min, x_max = X_train_2.iloc[:,0].min()-0.5, X_train_2.iloc[:,0].max()+0.5
y_min, y_max = X_train_2.iloc[:,1].min()-0.5, X_train_2.iloc[:,1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model_2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_boundary = go.Figure(data=[
    go.Contour(z=Z, x=np.arange(x_min, x_max, 0.02), y=np.arange(y_min, y_max, 0.02),
            colorscale='Blues', opacity=0.5, showscale=False),
    go.Scatter(x=X_train_2.iloc[:,0], y=X_train_2.iloc[:,1], mode='markers', 
            marker=dict(color=y_train, colorscale='Viridis', size=8),
                name="Training Data")
])
fig_boundary.update_layout(title=f"SVM Decision Boundary ({top_features[0]} vs {top_features[1]})",
                        xaxis_title=top_features[0],
                        yaxis_title=top_features[1])
st.plotly_chart(fig_boundary, use_container_width=True)

# Prediction Interface
st.subheader("🌼 Try a Prediction")

sl = st.slider("Sepal Length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), float(X["sepal length (cm)"].mean()))
sw = st.slider("Sepal Width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()), float(X["sepal width (cm)"].mean()))
pl = st.slider("Petal Length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), float(X["petal length (cm)"].mean()))
pw = st.slider("Petal Width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()), float(X["petal width (cm)"].mean()))

if st.button("Predict Flower Species"):
    input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_cols)
    prediction = model.predict(input_data)
    species = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🌸 Predicted Species: **{species}**")
