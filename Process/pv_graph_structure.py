import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from tqdm import tqdm

# Load data from CSV file
df = pd.read_csv("uspvdb.csv")

# Extract coordinates for KD-tree (latitude and longitude)
coords = df[['ylat', 'xlong']].values

# Create KD-tree for efficient neighbor search
tree = cKDTree(coords)

# Set distance threshold (50 km, convert to degrees approximately)
distance_threshold = 50 / 111  # Approximate conversion for 50 km to degrees

# Create an empty graph
G = nx.Graph()

# Add nodes (photovoltaic sites) with 'case_id' as the node ID and coordinates as attributes
for _, row in df.iterrows():
    G.add_node(row['case_id'], pos=(row['xlong'], row['ylat']))

# Prepare an empty list to store edge information for the text file
edges_info = []

# Find neighbors within threshold and add edges with tqdm progress bar
for i, (lat, lon) in tqdm(enumerate(coords), total=len(coords), desc="Processing nodes"):
    # Find neighbors within the distance threshold
    neighbors = tree.query_ball_point([lat, lon], distance_threshold)
    for j in neighbors:
        if i < j:  # To avoid duplicate edges
            dist = geodesic((lat, lon), coords[j]).km
            if dist <= 50:  # Only add edges within 50 km
                # Add edge to the graph
                G.add_edge(df.iloc[i]['case_id'], df.iloc[j]['case_id'], weight=dist)
                # Append edge information to list
                edges_info.append(f"{df.iloc[i]['case_id']} -- {df.iloc[j]['case_id']} [distance: {dist:.2f} km]")

# Save edge information to a text file
with open("C:/Users/15173/Desktop/EC601/Bohan1Zhang-GNN/edge_info.txt", "w") as f:
    f.write("\n".join(edges_info))

# Visualization of the graph
plt.figure(figsize=(10, 8))

# Get node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes (photovoltaic sites)
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='skyblue', edgecolors='k', label='Photovoltaic Sites')

# Draw edges
edges = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edges(G, pos, edgelist=edges.keys(), width=0.5, alpha=0.6, edge_color='gray')

# Add legend and title
plt.title("Distribution of Photovoltaic Sites and Connectivity", fontsize=15)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend(loc="upper right")

# Save the plot to a file
plt.savefig("C:/Users/15173/Desktop/EC601/Bohan1Zhang-GNN/photovoltaic_sites_distribution.png")

# Show plot
plt.show()
