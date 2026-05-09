# %%
##########################################################
##### ECHO CHAMBER SCORE ANALYSIS ########################
##########################################################

"""
Replication note
----------------
This script calculates Echo Chamber Scores (ECS) for the retweet networks
created in s3_network_analysis.py and the user-level embeddings created in
s4_ideology_detection.py.

The ECS implementation builds on the project:

"Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach"

by Faisal Alatawi, Paras Sheth, and Huan Liu.

GitHub repository:
https://github.com/faalatawi/echo-chamber-score

Expected folder structure:

    results/period_1/
        edges.csv
        nodes.csv
        embeddings.feather

    results/period_2/
        edges.csv
        nodes.csv
        embeddings.feather

    results/all_periods/
        edges.csv
        nodes.csv
        embeddings.feather

The helper files required for ECS calculation must be available in:

    src/load_data.py
    src/EchoGAE.py
    src/echo_chamber_measure.py
"""

# %%
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.utils import check_random_state

from src.load_data import get_data
from src.EchoGAE import EchoGAE_algorithm
from src.echo_chamber_measure import EchoChamberMeasure


# %%
# Set seed
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

check_random_state(seed_value)


# %%
# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"

PERIOD_1_PATH = PROJECT_DIR / "results" / "period_1"
PERIOD_2_PATH = PROJECT_DIR / "results" / "period_2"
ALL_PERIODS_PATH = PROJECT_DIR / "results" / "all_periods"

PERIOD_1_PATH.mkdir(parents=True, exist_ok=True)
PERIOD_2_PATH.mkdir(parents=True, exist_ok=True)
ALL_PERIODS_PATH.mkdir(parents=True, exist_ok=True)


# %%
# Choose dataset
# NOTE: Calculate the ECS for each period individually by selecting the relevant path.

# Period 1
# ds = PERIOD_1_PATH

# Period 2
# ds = PERIOD_2_PATH

# All periods
ds = ALL_PERIODS_PATH


# %%
# Filters
RES_VALUE = 0.1     # Resolution value for community detection
COMM_SIZE = 3       # Minimum number of members in each community
MIN_DEGREE = 0      # Minimum degree each node should have in the RT network
MIN_TWEETS = 2      # Minimum number of tweets posted in given period


# %%
# ECS metric
ecs_information = []

print(f"Dataset ({ds}): ", end="")

ds_dict = {}
ds_dict["dataset"] = str(ds)

# Get the data
# NOTE: See https://github.com/pyg-team/pytorch_geometric/issues/92
G, users_embeddings, community_labels, author_id_to_index_map, users_information = get_data(
    path=str(ds),
    resolution=RES_VALUE,
    min_comm_size=COMM_SIZE,
    directed_graph=True,
    min_degree=MIN_DEGREE,
    min_tweets=MIN_TWEETS
)


# %%
# Graph information
ds_dict["number_of_nodes"] = G.number_of_nodes()
ds_dict["number_of_edges"] = G.number_of_edges()
ds_dict["number_of_communities"] = len(np.unique(community_labels))


# %%
# ECS
user_emb = EchoGAE_algorithm(
    G,
    user_embeddings=users_embeddings,
    show_progress=False,
    hidden_channels=20,
    out_channels=10,
    epochs=300
)

ecm = EchoChamberMeasure(user_emb, community_labels)
eci = ecm.echo_chamber_index()

ds_dict["echo_chamber_score"] = eci

print(f"ECS = {eci:.3f} -- ", end=" ")


# %%
# Community ECIs and sizes
sizes = []
ECSs = []

for i in np.unique(community_labels):
    sizes.append(np.sum(community_labels == i))
    ECSs.append(ecm.community_echo_chamber_index(i))

ds_dict["community_sizes"] = sizes
ds_dict["community_ECIs"] = ECSs

print("")


# %%
# Create df for "ecs_information"
ecs_information.append(ds_dict)
ecs_information_df = pd.DataFrame(ecs_information)

# Delete from df "tweets" and "embeddings"
users_information = users_information[["author_id", "community"]]

# Save results
ecs_information_df.to_csv(ds / f"ecs_information_resolution_{RES_VALUE}.csv", index=False)
users_information.to_csv(ds / "users_information.csv", index=False)


# %%
# Create mappings from community to users and user to community
community_to_author_ids_map = {}
author_id_to_community_map = {}

# Extract all "author_ids" from the "author_id_index_map"
author_ids = author_id_to_index_map.keys()

# Iterate through pairs of "author_ids" and corresponding "community_labels"
for author_id, community_label in zip(author_ids, community_labels):
    community_to_author_ids_map.setdefault(f"Community_{community_label}", []).append(author_id)
    author_id_to_community_map[str(author_id)] = f"Community_{community_label}"

# Save "community_to_author_ids_map"
with open(ds / "community_to_author_ids_map.json", "w") as json_file:
    json.dump(community_to_author_ids_map, json_file)

# Save "author_id_to_community_map"
with open(ds / "author_id_to_community_map.json", "w") as json_file:
    json.dump(author_id_to_community_map, json_file)

print("\n\n")


# %%
##########################################################
##### MERGE COMMUNITY INFORMATION WITH TWEET DATA ########
##########################################################

# Load tweet data
tweets_df = pd.read_parquet(DATA_PATH / "tweets_cleaned.parquet")

required_columns = ["user_id"]
missing_columns = [col for col in required_columns if col not in tweets_df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}. "
        "Please make sure tweets_cleaned.parquet was produced by s1_tweet_cleaner.py."
    )

tweets_df["user_id"] = tweets_df["user_id"].astype(str)


# %%
# Merge community information: Period 2
users_information_df_p2 = pd.read_csv(PERIOD_2_PATH / "users_information.csv")

# Rename author_id to user_id for merging with tweet data
users_information_df_p2 = users_information_df_p2.rename(columns={"author_id": "user_id"})

# Convert user_id in both DataFrames to the same data type
users_information_df_p2["user_id"] = users_information_df_p2["user_id"].astype(str)

# Merge the DataFrames on user_id
merged_df = pd.merge(
    tweets_df,
    users_information_df_p2[["user_id", "community"]],
    on="user_id",
    how="left"
)

# Rename community column
merged_df.rename(columns={"community": "community_period_2"}, inplace=True)

# Change values to nullable integer while preserving NaN values
merged_df["community_period_2"] = merged_df["community_period_2"].astype("Int64")

# Get the distribution
community_distribution = merged_df["community_period_2"].value_counts(dropna=False)

# Print the distribution
print(community_distribution)


# %%
# Merge community information: All periods
users_information_df_all = pd.read_csv(ALL_PERIODS_PATH / "users_information.csv")

# Rename author_id to user_id for merging with tweet data
users_information_df_all = users_information_df_all.rename(columns={"author_id": "user_id"})

# Convert user_id in both DataFrames to the same data type
users_information_df_all["user_id"] = users_information_df_all["user_id"].astype(str)

# Merge the DataFrames on user_id
merged_df = pd.merge(
    merged_df,
    users_information_df_all[["user_id", "community"]],
    on="user_id",
    how="left"
)

# Rename community column
merged_df.rename(columns={"community": "community_all_periods"}, inplace=True)

# Change values to nullable integer while preserving NaN values
merged_df["community_all_periods"] = merged_df["community_all_periods"].astype("Int64")

# Get the distribution
community_distribution = merged_df["community_all_periods"].value_counts(dropna=False)

# Print the distribution
print(community_distribution)


# %%
# Save the DataFrame
merged_df.to_parquet(DATA_PATH / "tweets_cleaned_community.parquet", index=False)

# %%
