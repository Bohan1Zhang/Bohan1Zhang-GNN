import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
from itertools import combinations

# Load your data (replace with the correct file path)
file_path = '/Users/harlem/Bohan1Zhang-GNN/Bohan1Zhang-GNN/Bohan1Zhang-GNN/Bohan1Zhang-GNN/uspvdb.csv'
df = pd.read_csv(file_path)

# Filter for Massachusetts (MA) data and select columns
ma_data = df[df['p_state'] == 'MA'][['case_id', 'ylat', 'xlong', 'p_cap_dc']]

# Extract relevant information for GNN
locations_capacities = list(zip(ma_data['case_id'], ma_data['ylat'], ma_data['xlong'], ma_data['p_cap_dc']))

# Define a distance threshold (20 km)
distance_threshold = 20
close_pairs = []

# Calculate distances and find pairs within 20 km, compute average p_cap_dc
for (case_id1, lat1, long1, cap1), (case_id2, lat2, long2, cap2) in combinations(locations_capacities, 2):
    distance = geodesic((lat1, long1), (lat2, long2)).km
    if distance <= distance_threshold:
        avg_cap = (cap1 + cap2) / 2
        close_pairs.append((case_id1, case_id2, distance, avg_cap))

# Create a DataFrame for edge data
close_pairs_df = pd.DataFrame(close_pairs, columns=['case_id_1', 'case_id_2', 'distance_km', 'average_p_cap_dc'])

# Create a graph
G = nx.Graph()
for _, row in ma_data.iterrows():
    G.add_node(row['case_id'], pos=(row['xlong'], row['ylat']))

for _, row in close_pairs_df.iterrows():
    G.add_edge(row['case_id_1'], row['case_id_2'], weight=row['average_p_cap_dc'])

# Sort weights and divide them into five levels
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

# Assign colors to edges based on their weight partition
edge_colors = [weight_to_color[weight] for weight in weights]

# Draw the graph
plt.figure(figsize=(18, 18))

# Draw nodes
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.8)

# Draw edges with color coding
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

# Remove axis for a cleaner view
plt.axis("off")

output_path = 'solar_plant_network_partitioned_avg_cap.png'
plt.savefig(output_path, format='png')

