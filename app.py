import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Load models
@st.cache_resource
def load_models():
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    kmeans = pickle.load(open('kmeans.pkl', 'rb'))
    encoder = load_model('encoder.h5')
    return scaler, kmeans, encoder

scaler, kmeans, encoder = load_models()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('rfm_data.csv')

rfm = load_data()

# Compute latent features
@st.cache_data
def compute_latent(data):
    scaled = scaler.transform(data[['Recency', 'Frequency', 'Monetary']])
    return encoder.predict(scaled)

latent_all = compute_latent(rfm)

# Cluster labels (adjust based on your analysis)
cluster_labels = {
    0: "Low Value Customers",
    1: "Churn Risk Customers",
    2: "Regular Customers",
    3: "High Value / VIP Customers"
}

# UI
st.title("Retail Customer Segmentation System")

st.sidebar.header("Customer Input")

recency = st.sidebar.number_input("Recency", 0, 365, 30)
frequency = st.sidebar.number_input("Frequency", 1, 500, 10)
monetary = st.sidebar.number_input("Monetary", 0.0, 10000.0, 500.0)

# Prediction
if st.sidebar.button("Predict"):
    input_data = np.array([[recency, frequency, monetary]])
    
    scaled = scaler.transform(input_data)
    latent = encoder.predict(scaled)
    cluster = kmeans.predict(latent)[0]

    segment = cluster_labels.get(cluster, "Unknown")
    st.subheader(f"Predicted Segment: {segment}")

    # Similar customers
    similarity = cosine_similarity(latent_all, latent)
    top_idx = similarity.flatten().argsort()[::-1][1:6]

    st.subheader("Similar Customers")
    st.dataframe(rfm.iloc[top_idx])

# Optional cluster summary
if st.checkbox("Show Cluster Summary"):
    st.dataframe(rfm.groupby('Cluster').mean())