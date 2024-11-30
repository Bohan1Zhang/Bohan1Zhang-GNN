import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import os

class SolarGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(SolarGCN, self).__init__()
        self.hidden_channels = hidden_channels  
        
        # Convolution layers
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels // 2)
        self.conv4 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels // 4)
        self.fc = nn.Linear(hidden_channels // 4, 2)
        
        # Skip connections
        self.skip1 = nn.Linear(num_features, hidden_channels)
        self.skip2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.skip3 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        
    def forward(self, x, edge_index, edge_weight):
        identity = x
        
        # First layer with skip connection
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x + self.skip1(identity))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second layer with skip connection
        identity2 = x
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x + identity2)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Third layer with skip connection
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x + self.skip2(identity2[:, :self.hidden_channels]))
        
        # Fourth layer with skip connection
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Final layer
        x = self.fc(x)
        return x

def normalize_coordinates(coords, ma_bounds):
    """Normalize coordinates to [0,1] range based on MA bounds"""
    lat_min, lat_max = ma_bounds[0]
    lon_min, lon_max = ma_bounds[1]
    
    normalized_coords = np.zeros_like(coords)
    normalized_coords[:, 0] = (coords[:, 0] - lat_min) / (lat_max - lat_min)
    normalized_coords[:, 1] = (coords[:, 1] - lon_min) / (lon_max - lon_min)
    
    return normalized_coords

def denormalize_coordinates(normalized_coords, ma_bounds):
    """Convert normalized coordinates back to actual values"""
    lat_min, lat_max = ma_bounds[0]
    lon_min, lon_max = ma_bounds[1]
    
    coords = np.zeros_like(normalized_coords)
    coords[:, 0] = normalized_coords[:, 0] * (lat_max - lat_min) + lat_min
    coords[:, 1] = normalized_coords[:, 1] * (lon_max - lon_min) + lon_min
    
    return coords

def prepare_solar_data(file_path, distance_threshold=20):
    print("Loading data...")
    df = pd.read_csv(file_path)
    ma_data = df[df['p_state'] == 'MA'][['case_id', 'ylat', 'xlong', 'p_cap_dc']].reset_index(drop=True)
    print(f"Found {len(ma_data)} solar plants in Massachusetts")
    
    # MA bounds for coordinate normalization
    ma_bounds = np.array([
        [41.2, 43.0],  # latitude bounds
        [-74.0, -69.5]  # longitude bounds
    ])
    
    # Normalize coordinates
    coords = ma_data[['ylat', 'xlong']].values
    normalized_coords = normalize_coordinates(coords, ma_bounds)
    
    # Prepare features
    scaler = StandardScaler()
    node_features = np.column_stack([
        normalized_coords,
        scaler.fit_transform(ma_data[['p_cap_dc']].values)
    ])
    
    print("Creating graph edges...")
    edges = []
    edge_weights = []
    locations = list(enumerate(zip(ma_data['ylat'], ma_data['xlong'])))
    
    for (idx1, (lat1, long1)), (idx2, (lat2, long2)) in combinations(locations, 2):
        distance = geodesic((lat1, long1), (lat2, long2)).km
        if distance <= distance_threshold:
            edges.append([idx1, idx2])
            edge_weights.append(1.0 / (distance + 1e-6))
    
    print(f"Created {len(edges)} edges between solar plants")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    edge_index_reversed = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_reversed], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight])
    
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_weight=edge_weight,
        pos=torch.tensor(normalized_coords, dtype=torch.float)
    )
    
    return data, scaler, ma_data, ma_bounds

def train_solar_gcn(data, model, ma_bounds, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=20)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    # Convert bounds to tensor for loss calculation
    bounds_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.float)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        
        # MSE loss for position prediction
        mse_loss = F.mse_loss(out, data.pos)
        
        # Boundary loss using sigmoid
        out_normalized = torch.sigmoid(out)  # Ensure predictions are in [0,1]
        boundary_loss = F.mse_loss(out_normalized, out)
        
        # Additional penalty for predictions outside [0,1]
        out_of_bounds = torch.sum(F.relu(torch.abs(out_normalized - 0.5) - 0.5)) * 100.0
        
        # Total loss
        loss = mse_loss + boundary_loss + out_of_bounds
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}, '
                  f'MSE: {mse_loss.item():.4f}, Boundary: {boundary_loss.item():.4f}, '
                  f'Out of Bounds: {out_of_bounds.item():.4f}')
        
        if loss < best_loss:
            best_loss = loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return model

def predict_new_locations(model, data, ma_bounds, num_predictions=10):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_weight)
        normalized_predictions = torch.sigmoid(predictions).numpy()
        
        # Denormalize predictions
        predictions = denormalize_coordinates(normalized_predictions, ma_bounds)
        
        # Select diverse predictions
        selected_predictions = []
        min_distance = 5
        
        # Shuffle predictions
        indices = np.random.permutation(len(predictions))
        
        for idx in indices:
            point = predictions[idx]
            if (41.2 <= point[0] <= 43.0 and -74.0 <= point[1] <= -69.5):
                if not selected_predictions or all(
                    geodesic(point, p).km > min_distance for p in selected_predictions
                ):
                    selected_predictions.append(point)
                    if len(selected_predictions) == num_predictions:
                        break
        
        if len(selected_predictions) < num_predictions:
            print(f"Warning: Only found {len(selected_predictions)} valid locations")
        
        if not selected_predictions:
            print("Warning: No valid points found within Massachusetts bounds")
            print("Range of predictions:", 
                  np.min(predictions, axis=0),
                  np.max(predictions, axis=0))
            return np.array([]).reshape(0, 2)
        
        return np.array(selected_predictions)

def visualize_solar_plants(existing_data, predicted_locations=None, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    geometry = [Point(xy) for xy in zip(existing_data['xlong'], existing_data['ylat'])]
    gdf = gpd.GeoDataFrame(existing_data, geometry=geometry, crs="EPSG:4326")
    
    ax1.set_title('Existing Solar Plants in Massachusetts', fontsize=14)
    gdf.plot(ax=ax1, color='red', markersize=50, alpha=0.6, label='Existing Plants')
    ctx.add_basemap(ax1, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax2.set_title('Existing and Predicted Solar Plants', fontsize=14)
    gdf.plot(ax=ax2, color='red', markersize=50, alpha=0.6, label='Existing Plants')
    
    if predicted_locations is not None and len(predicted_locations) > 0:
        try:
            if len(predicted_locations.shape) == 1:
                predicted_locations = predicted_locations.reshape(-1, 2)
            
            pred_geometry = [Point(lon, lat) for lat, lon in predicted_locations]
            pred_gdf = gpd.GeoDataFrame(geometry=pred_geometry, crs="EPSG:4326")
            pred_gdf.plot(ax=ax2, color='blue', markersize=50, alpha=0.6, label='Predicted Plants')
            
            for idx, row in pred_gdf.iterrows():
                coord = f"({row.geometry.y:.2f}, {row.geometry.x:.2f})"
                ax2.annotate(coord, (row.geometry.x, row.geometry.y), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='blue')
                
        except Exception as e:
            print(f"Warning: Error plotting predictions - {str(e)}")
            print(f"Shape of predictions array: {predicted_locations.shape}")
            print(f"First few predictions: {predicted_locations[:5]}")
    
    ctx.add_basemap(ax2, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    file_path = '/Users/harlem/Desktop/GNN/Bohan1Zhang-GNN-main/uspvdb.csv'
    
    print("Starting solar plant location prediction...")
    print("Loading data from:", file_path)
    
    data, scaler, ma_data, ma_bounds = prepare_solar_data(file_path)
    
    model = SolarGCN(num_features=3, hidden_channels=128)
    
    print("\nStarting model training...")
    trained_model = train_solar_gcn(data, model, ma_bounds)
    
    print("\nGenerating predictions...")
    predictions = predict_new_locations(trained_model, data, ma_bounds, num_predictions=10)
    
    if len(predictions) > 0:
        print("\nCreating visualizations...")
        output_path = os.path.join(os.path.dirname(file_path), 'solar_plants_prediction.png')
        visualize_solar_plants(ma_data, predictions, save_path=output_path)
        
        pred_df = pd.DataFrame(predictions, columns=['Latitude', 'Longitude'])
        print("\nPredicted new solar plant locations:")
        print(pred_df)
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("\nNo valid predictions were generated. Try adjusting the model parameters.")

if __name__ == "__main__":
    main()