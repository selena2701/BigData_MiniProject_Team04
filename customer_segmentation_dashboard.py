import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV/XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data")
    # Convert object columns to string to avoid Arrow serialization issues
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)
    st.write(df.head(100))

    # Sidebar for parameters
    st.sidebar.header("Clustering Settings")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.sidebar.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)
    n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)

    if selected_features:
        # Data cleaning
        df_clean = df[selected_features].copy()
        
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median
        df_clean = df_clean.fillna(df_clean.median())
        
        # Show data quality metrics
        st.subheader("Data Quality Metrics")
        missing_values = df_clean.isnull().sum()
        st.write("Missing values after cleaning:", missing_values)
        
        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # Run KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = labels

        st.subheader("Clustered Data")
        # Convert object columns to string to avoid Arrow serialization issues
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
        st.write(df.head(100))

        # Silhouette Score
        score = silhouette_score(X_scaled, labels)
        st.write(f"Silhouette Score: {score:.2f}")

        # Plot clusters (if at least 2 features)
        if len(selected_features) >= 2:
            fig = px.scatter(
                df,
                x=selected_features[0],
                y=selected_features[1],
                color=df['Cluster'].astype(str),
                title=f"Clusters on {selected_features[0]} vs {selected_features[1]}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show cluster centers
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)
        st.subheader("Cluster Centers")
        st.write(centers)
    else:
        st.warning("Please select at least one numeric feature for clustering.")
else:
    st.info("Please upload a CSV or XLSX file to start.")
