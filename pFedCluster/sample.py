import matplotlib.pyplot as plt
import numpy as np

# Increase the number of clients
num_clients = 100

# Generate random positions for clients
np.random.seed(0)
x = np.random.rand(num_clients)
y = np.random.rand(num_clients)

# Assign colors to the clients
colors = np.array(['red', 'green', 'blue'] * (num_clients // 3) + ['red'] * (num_clients % 3))

# First plot: clients evenly distributed without axes
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.scatter(x, y, c=colors, s=50, alpha=0.7)
plt.text(0.5, 1.05, "Evenly Distributed Clients", ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.axis('equal')
plt.axis('off')

# Second plot: clients clustered by color without axes
plt.subplot(1, 2, 2)
cluster_centers = {
    'red': (0.2, 0.8),
    'green': (0.5, 0.2),
    'blue': (0.8, 0.8)
}

for color in ['red', 'green', 'blue']:
    mask = colors == color
    x_color = x[mask]
    y_color = y[mask]

    # Cluster the clients by shifting their positions
    if color == 'red':
        x_color = 0.1 + 0.2 * x_color
        y_color = 0.7 + 0.2 * y_color
    elif color == 'green':
        x_color = 0.4 + 0.2 * x_color
        y_color = 0.1 + 0.2 * y_color
    elif color == 'blue':
        x_color = 0.7 + 0.2 * x_color
        y_color = 0.7 + 0.2 * y_color

    plt.scatter(x_color, y_color, c=color, s=50, alpha=0.7)
    center = cluster_centers[color]
    plt.scatter(center[0], center[1], c=color, s=200, alpha=0.3, edgecolors='black')

# Calculate and plot distances within and between clusters
from itertools import combinations


def calculate_distances(x, y, colors, cluster_centers):
    intra_cluster_distances = {}
    inter_cluster_distances = {}

    for color in cluster_centers:
        mask = colors == color
        x_color = x[mask]
        y_color = y[mask]
        distances = np.sqrt((x_color - x_color[:, None]) ** 2 + (y_color - y_color[:, None]) ** 2)
        intra_cluster_distances[color] = np.mean(distances[np.triu_indices(len(x_color), k=1)])

    for (color1, center1), (color2, center2) in combinations(cluster_centers.items(), 2):
        inter_cluster_distances[(color1, color2)] = np.linalg.norm(np.array(center1) - np.array(center2))

    return intra_cluster_distances, inter_cluster_distances


intra_cluster_distances, inter_cluster_distances = calculate_distances(x, y, colors, cluster_centers)

# Draw lines between cluster centers to show inter-cluster distance
for (color1, color2), distance in inter_cluster_distances.items():
    center1 = cluster_centers[color1]
    center2 = cluster_centers[color2]
    plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 'k--', alpha=0.5)
    plt.text((center1[0] + center2[0]) / 2, (center1[1] + center2[1]) / 2, f"{distance:.2f}", ha='center', va='center',
             fontsize=12, color='black')

# Annotate intra-cluster distances
for color, distance in intra_cluster_distances.items():
    center = cluster_centers[color]
    plt.text(center[0], center[1] - 0.1, f"Intra: {distance:.2f}", ha='center', va='center', fontsize=12, color='black')

plt.text(0.5, 1.05, "Clustered Clients", ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.axis('equal')
plt.axis('off')

plt.tight_layout()
plt.show()
