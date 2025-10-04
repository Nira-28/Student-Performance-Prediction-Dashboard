import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# ---------------- Load Data ----------------
data = pd.read_csv("AI-Data.csv")
data = shuffle(data)

st.title("Student Performance Prediction Dashboard")
st.subheader("Dataset Overview")
st.dataframe(data.head())

# ---------------- Correlation Heatmap ----------------
st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(10,6))
sb.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ---------------- Feature Selection ----------------
drop_cols = ["gender","StageID","GradeID","NationalITy","PlaceofBirth","SectionID",
             "Topic","Semester","Relation","ParentschoolSatisfaction",
             "ParentAnsweringSurvey","AnnouncementsView"]
clean_data = data.drop(columns=drop_cols)

# Encode categorical
for col in clean_data.columns:
    if clean_data[col].dtype == object:
        le = LabelEncoder()
        clean_data[col] = le.fit_transform(clean_data[col])

# Split data
ind = int(len(clean_data) * 0.7)
X = clean_data.values[:,0:4]
y = clean_data.values[:,4]
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]

# ---------------- Train Models ----------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Perceptron": Perceptron(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "MLP Classifier": MLPClassifier(activation="logistic", max_iter=500)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = sum(pred==y_test)/len(y_test)
    results[name] = (model, acc)
    st.write(f"**{name} Accuracy:** {round(acc,3)}")
    st.text(classification_report(y_test, pred))

# ---------------- User Input for Prediction ----------------
st.subheader("Test Your Own Input")
rai = st.number_input("Raised Hands", min_value=0, max_value=50, value=5)
res = st.number_input("Visited Resources", min_value=0, max_value=50, value=5)
dis = st.number_input("Discussions Count", min_value=0, max_value=50, value=5)
absc = st.selectbox("Number of Absences", ["Under-7", "Above-7"])
absc_val = 1 if absc == "Under-7" else 0

user_input = np.array([rai, res, dis, absc_val]).reshape(1,-1)

if st.button("Predict"):
    st.subheader("Predictions:")
    for name, (model, acc) in results.items():
        pred_class = model.predict(user_input)[0]
        pred_label = {0:"H",1:"M",2:"L"}.get(pred_class, pred_class)
        st.write(f"{name}: {pred_label}")
