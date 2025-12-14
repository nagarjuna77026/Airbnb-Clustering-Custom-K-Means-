# Airbnb Listing Clustering – K-Means Machine Learning Project

## 📌 Overview
This project applies unsupervised machine learning techniques to cluster Airbnb listings based on pricing, availability, and accommodation features. The goal is to identify meaningful groups of listings that can support market analysis, pricing strategies, and customer segmentation.

---

## 🎯 Problem Statement
Group Airbnb listings into distinct clusters using K-Means clustering to uncover hidden patterns in listing characteristics.

- **Input:** Airbnb listings dataset
- **Objective:** Identify optimal clusters and interpret listing segments
- **Approach:** Feature preprocessing, clustering, and visualization

---

## 📂 Dataset
- Airbnb listings dataset (CSV format)
- Contains information such as price, location, room type, availability, and review metrics
- Loaded and processed within `airbnb_clustering.ipynb`

---

## ⚙️ Data Preprocessing
The following preprocessing steps are performed:

- Inspected data types and missing values
- Removed irrelevant or identifier columns
- Handled missing values using appropriate imputation
- Encoded categorical features (e.g., room type, neighborhood)
- Scaled numerical features to ensure equal contribution in distance-based clustering

---

## 🧠 Feature Selection
Key features used for clustering include:
- Price
- Availability
- Minimum nights
- Number of reviews
- Review scores
- Room type (encoded)

---

## 🔍 Clustering Methodology
- Algorithm: **K-Means Clustering**
- Distance Metric: Euclidean distance
- Optimal number of clusters determined using:
  - Elbow Method
  - Inertia analysis

---

## 📊 Model Evaluation
Since clustering is unsupervised, evaluation is based on:
- Inertia (Within-Cluster Sum of Squares)
- Visual inspection of cluster separation
- Interpretation of cluster centroids

---

## 📈 Visualizations
The project includes:
- Elbow curve to determine optimal K value
- Scatter plots showing clustered listings
- Cluster centroid visualizations
- Feature-wise cluster comparison plots

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd airbnb-clustering
2. Install dependencies:
   pip install -r requirements.txt
3. Run the notebook:
   jupyter notebook airbnb_clustering.ipynb
