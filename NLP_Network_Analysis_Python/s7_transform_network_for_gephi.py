# %%
#########################################
### TRANSFORM NETWORK DATA FOR GEPHI ###
#########################################

"""
Replication note
----------------
This script transforms retweet network data and ECS community assignments
into node and edge tables that can be directly imported into Gephi.

Expected input:

    results/all_periods/
        edges.csv
        users_information.csv

Outputs:

    results/all_periods/
        nodes_gephi.csv
        edges_gephi.csv
"""

import pandas as pd

from pathlib import Path


# %%
# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Define path
# DATA_PATH = PROJECT_DIR / "results" / "period_1"
# DATA_PATH = PROJECT_DIR / "results" / "period_2"

DATA_PATH = PROJECT_DIR / "results" / "all_periods"


# %%
# Load data
edges_init = pd.read_csv(DATA_PATH / "edges.csv")
nodes_ecs = pd.read_csv(DATA_PATH / "users_information.csv")


# %%
# Transform edges and nodes

# Keep only edges between users contained in ECS results
edges_gephi_tmp = edges_init[
    (edges_init["source"].isin(nodes_ecs["author_id"])) &
    (edges_init["target"].isin(nodes_ecs["author_id"]))
]

# Create nodes table
nodes_gephi = pd.DataFrame({
    "label": nodes_ecs["author_id"].tolist()
})

nodes_gephi.insert(
    0,
    "id",
    range(1, len(nodes_ecs["author_id"].tolist()) + 1)
)

nodes_gephi.insert(
    2,
    "community",
    nodes_ecs["community"].tolist()
)

# Create label-to-id mapping
label_to_id = dict(zip(nodes_gephi["label"], nodes_gephi["id"]))

# Create edges table
edges_gephi = edges_gephi_tmp.copy()

edges_gephi["source"] = edges_gephi_tmp["source"].map(label_to_id)
edges_gephi["target"] = edges_gephi_tmp["target"].map(label_to_id)


# %%
# Save results
nodes_gephi.to_csv(DATA_PATH / "nodes_gephi.csv", index=False)
edges_gephi.to_csv(DATA_PATH / "edges_gephi.csv", index=False)

# %%
