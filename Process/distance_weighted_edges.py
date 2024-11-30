import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
import numpy as np
from itertools import combinations  # Add this line for combinations

# Load your data (replace with the correct file path)
file_path = '/Users/harlem/Bohan1Zhang-GNN/Bohan1Zhang-GNN/Bohan1Zhang-GNN/uspvdb.csv'
df = pd.read_csv(file_path)

# Filter for Massachusetts (MA) data
ma_data = df[df['p_state'] == 'MA']

# Extract relevant location information for GNN
locations = list(zip(ma_data['case_id'], ma_data['ylat'], ma_data['xlong']))

# Define a distance threshold (20 km in this case)
distance_threshold = 20
close_pairs = []
weights = []

# Calculate distances and generate edges with weights
epsilon = 0.1  # Small value to avoid division by zero
for (case_id1, lat1, long1), (case_id2, lat2, long2) in combinations(locations, 2):
    distance = geodesic((lat1, long1), (lat2, long2)).km
    if distance <= distance_threshold:
        close_pairs.append((case_id1, case_id2))
        weights.append(1 / (distance + epsilon))  # Weight inversely proportional to distance

# Create a DataFrame for edges and weights
close_pairs_df = pd.DataFrame(close_pairs, columns=['case_id_1', 'case_id_2'])
close_pairs_df['weight'] = weights

# Create a graph for visualization
G = nx.Graph()
nodes = list(set(close_pairs_df['case_id_1']).union(set(close_pairs_df['case_id_2'])))
node_positions = {case_id: (row['xlong'], row['ylat']) for case_id, row in ma_data.set_index('case_id').loc[nodes].iterrows()}

for node, pos in node_positions.items():
    G.add_node(node, pos=pos)

for _, row in close_pairs_df.iterrows():
    G.add_edge(row['case_id_1'], row['case_id_2'], weight=row['weight'])

# Sort weights and divide into five partitions
weights = [G[u][v]['weight'] for u, v in G.edges()]
sorted_weights = sorted(weights)
n = len(sorted_weights)
partitions = [
    sorted_weights[:n // 5],                          # Lowest 20%
    sorted_weights[n // 5:2 * n // 5],                 # Next 20%
    sorted_weights[2 * n // 5:3 * n // 5],             # Middle 20%
    sorted_weights[3 * n // 5:4 * n // 5],             # Next 20%
    sorted_weights[4 * n // 5:]                       # Highest 20%
]

# Define colors for partitions
partition_colors = {
    0: 'purple',   # Lowest 20%
    1: 'green',    # Next 20%
    2: 'yellow',   # Middle 20%
    3: 'orange',   # Next 20%
    4: 'red'       # Highest 20%
}

# Map weights to colors
weight_to_color = {}
for i, partition in enumerate(partitions):
    for weight in partition:
        weight_to_color[weight] = partition_colors[i]

edge_colors = [weight_to_color[weight] for weight in weights]

# Draw the graph
plt.figure(figsize=(18, 18))

# Draw nodes and edges with mapped colors
nx.draw_networkx_nodes(G, node_positions, node_size=30, node_color='black', alpha=0.8)
nx.draw_networkx_edges(G, node_positions, edge_color=edge_colors, width=1)

# Remove axis for clarity
plt.axis("off")

output_path = 'solar_plant_network_visualization_partitioned.png'
plt.savefig(output_path, format='png')


