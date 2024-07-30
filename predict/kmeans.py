import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from process import create_x_data_conv2d

# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
data_dir = "data/conv2d.csv"
data_reshaped_save_dir = "data/clustered_data_7d.csv"
model_dir = "model/kmeans_model.pkl"

# 원본 데이터
df = pd.read_csv(data_dir)

# 변수 설정
data_cnt: int = 120000
test_cnt: int = 120
x_days: int = 7
x_cols: list = ["volume_ratio", "down_delta", "delta", "up_delta"]
y_cols: list = ["down_delta", "delta", "up_delta"]
cluster_num: int = 0

df = df[-data_cnt:]

# x_data 데이터 생성
data = create_x_data_conv2d(df, x_cols, x_days, 1)
data_reshaped = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

# K-Means 클러스터링 수행 (K=3)
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(data_reshaped)

# 클러스터 할당 결과
labels = kmeans.labels_

# K-Means 모델 저장
joblib.dump(kmeans, model_dir)
print("Model saved successfully!")

# 클러스터 할당 결과를 csv 파일로 저장
df_reshaped = pd.DataFrame(data_reshaped)
df_reshaped["Cluster"] = labels
df_reshaped.to_csv(data_reshaped_save_dir, index=False)

# 마지막 3만개의 데이터에 대한 클러스터 할당 결과
subset_labels = labels[-30000:]

# 마지막 3만개의 데이터 (고차원 데이터를 시각화하기 위해 2D로 축소)
subset_data = data_reshaped[-30000:]

# 차원 축소를 위해 PCA 사용 (2D로 축소)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(subset_data)

# 시각화
plt.figure(figsize=(10, 7))
for i in range(3):
    cluster_points = reduced_data[subset_labels == i]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", alpha=0.5
    )

plt.title("K-Means Clustering (K=3) on Last 30000 Data Points")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 클러스터 할당 결과 배열 반환
print(labels[-100:])
