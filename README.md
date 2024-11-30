# GNN prediction model for PV siting

# Define of project
The development of renewable energy sources has become a must for the modern era, when traditional combustion energy sources are becoming scarce and expensive and fission energy sources are still dangerous.
 Among the renewable energy sources, Photovoltaic power sources stand out with lower environmental impact and lower maintenance difficulty. However, there are still problems to be solved if photovoltaic power
 plants should become the main source of power. Photovoltaic power relies on light to generate electricity, a widely used and necessary resource to multiple entities. Sunlight is also not constant, as it would
 change depending on conditions including but not limited to the air condition of the second, weather of the day, and day of the year. Therefore, choosing a good location for the photovoltaic power plant is
 vital. We would want to push forward the use of photovoltaic power sources by providing an AI that would give suggestions on where to build new photovoltaic power plants based on multiple considerations like
 the landscape, solar resource abundance, local population density and power usage, etc. We hope that our product Photo would target the pain point of not knowing where to plan the photovoltaic power plants like
 antibodies to antigen, and maximize the overall contribution of photovoltaic power plants to the entire environment.


# Instructions to Set Up and Run the GCN Model
Step 1: Clone the Repository

Step 2: Set Up the Virtual Environment
  -Create a new Conda environment: "conda create --name GNN_network_proj python=3.9 -y"
  
  -Activate the environment:"conda activate GNN_network_proj"
  
  -Install dependencies:
    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    "pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric pandas scikit-learn matplotlib"

   -Visualize Results:
    "pip install basemap"


Step 3: Prepare the Dataset
    Ensure the dataset (merged_data.csv) is located in the same directory as the GCN script (GCN.py). If not, provide the correct path in the code.

Step 4:  Run the Model
    python GCN.py

