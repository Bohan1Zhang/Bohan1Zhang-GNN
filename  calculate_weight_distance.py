import pandas as pd
from geopy.distance import geodesic
from itertools import combinations

# Load the original data file 
file_path = '/Users/harlem/Bohan1Zhang-GNN/Bohan1Zhang-GNN/Bohan1Zhang-GNN/uspvdb.csv'  
df = pd.read_csv(file_path)

# Filter for Massachusetts (MA) data
ma_data = df[df['p_state'] == 'MA']

# Prepare location data for calculating distances
locations = list(zip(ma_data['case_id'], ma_data['ylat'], ma_data['xlong']))

# Define a distance threshold (20 km) and epsilon for weight calculation
distance_threshold = 20  # kilometers
epsilon = 0.1
close_pairs = []

# Calculate distances and collect pairs within 20km
for (case_id1, lat1, long1), (case_id2, lat2, long2) in combinations(locations, 2):
    distance = geodesic((lat1, long1), (lat2, long2)).km
    if distance <= distance_threshold:
        weight = 1 / (distance + epsilon)
        close_pairs.append((case_id1, case_id2, round(distance, 1), round(weight, 3)))

# Convert pairs into a DataFrame
close_pairs_df = pd.DataFrame(close_pairs, columns=['case_id_1', 'case_id_2', 'distance_km', 'weight'])

# Save to CSV
output_path = 'Solar_Plant_Pairs_within_20km_with_Rounded_Weights_in_Massachusetts.csv'
close_pairs_df.to_csv(output_path, index=False)

print(f"Table saved to {output_path}")