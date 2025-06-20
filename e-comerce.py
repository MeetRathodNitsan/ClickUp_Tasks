
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold


np.random.seed(42)
n_customers = 200

data = {
    'customer_id': np.arange(1, n_customers + 1),
    'total_spent': np.random.normal(loc=500, scale=150, size=n_customers),
    'visits_per_month': np.random.poisson(lam=4, size=n_customers),
    'items_per_order': np.random.normal(loc=3, scale=1.0, size=n_customers),
    'days_since_last_purchase': np.random.exponential(scale=10, size=n_customers),
    'returns': np.random.binomial(n=1, p=0.1, size=n_customers),
    'days_since_last_visit': np.random.randint(1, 100, size=n_customers)
}

df = pd.DataFrame(data)

X = df.drop(columns=['customer_id'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='Set2')
plt.title('Customer Segmentation using KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()


df['spending_per_visit'] = df['total_spent'] / (df['visits_per_month'] + 1)


df['visit_frequency'] = pd.qcut(df['days_since_last_visit'], q=4, labels=False)


df['return_behaviour'] = df['returns'] * df['items_per_order']


print("🔧 Engineered Features (First 5 Rows):")
print(df[['spending_per_visit', 'visit_frequency', 'return_behaviour']].head())


X_features = df.drop(columns=['customer_id', 'cluster'])
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X_features)

print(f"\n✅ Selected {X_selected.shape[1]} features after removing low-variance features.")
