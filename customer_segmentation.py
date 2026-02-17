import pandas as pd
from sklearn.cluster import KMeans
df = pd.read_csv("../data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
df.to_csv("../outputs/task2_customer_segments.csv", index=False)
print("Task-2 completed. Clustered data saved.")