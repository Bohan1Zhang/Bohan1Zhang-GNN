import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Load data
data_path = 'C:\\Users\\15173\\Desktop\\EC601\\Bohan1Zhang-GNN\\Resault\\merged_data.csv'
df = pd.read_csv(data_path)

# Filter for MA state data
df = df[df['State'] == 'MA']

# Extract features, latitude/longitude
features = df.iloc[:, 4:-1].values  # Columns E to P
latlong = df[['ylat', 'xlong']].values  # Columns C (latitude) and D

# Standardize features and latitude/longitude
scaler_features = StandardScaler()
scaler_latlong = StandardScaler()
features = scaler_features.fit_transform(features)
latlong = scaler_latlong.fit_transform(latlong)

# Construct edge relationships (using KNN)
adj_matrix = kneighbors_graph(latlong, n_neighbors=5, mode='connectivity').toarray()
edges = np.array(np.nonzero(adj_matrix))
edge_index = torch.tensor(edges, dtype=torch.long)

# Create PyG data object
data = Data(
    x=torch.tensor(features, dtype=torch.float),  # Node features
    edge_index=edge_index,  # Edges
    y=torch.tensor(latlong, dtype=torch.float),  # Target lat/long
)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.x.size(1), 64, 2).to(device)  # Output dimension: 2 (lat/long)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(500):  # Train for 500 epochs
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out, data.y)  # MSE loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Predict new nodes
model.eval()
predicted_positions = model(data.x, data.edge_index).detach().cpu().numpy()

# Generate 10 predicted nodes
future_nodes = scaler_latlong.inverse_transform(predicted_positions[:10])  # Inverse transform to original scale
existing_nodes = scaler_latlong.inverse_transform(data.y.cpu().numpy())  # Existing nodes in original scale

# Visualize nodes on a U.S. map
fig = plt.figure(figsize=(12, 8))
m = Basemap(
    projection='merc',
    llcrnrlon=-73.5, urcrnrlon=-69.5,  # Longitude range for MA
    llcrnrlat=41, urcrnrlat=43,        # Latitude range for MA
    resolution='i'
)

m.drawcoastlines()
m.drawcountries()
m.drawstates()

# Convert lat/long to map coordinates
existing_x, existing_y = m(existing_nodes[:, 1], existing_nodes[:, 0])  # Existing nodes
future_x, future_y = m(future_nodes[:, 1], future_nodes[:, 0])  # Future nodes

# Plot existing and predicted nodes
m.scatter(existing_x, existing_y, s=10, label='Existing Nodes', color='blue', alpha=0.6)
m.scatter(future_x, future_y, s=30, label='Predicted Nodes', color='red', alpha=0.8)

plt.legend()
plt.title("Existing and Predicted Photovoltaic Sites in MA")
plt.show()
