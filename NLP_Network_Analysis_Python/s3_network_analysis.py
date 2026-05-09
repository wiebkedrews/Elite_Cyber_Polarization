# %%
########################
### NETWORK ANALYSIS ###
########################

"""
Replication note
----------------
This script creates retweet network edge and node lists for all tweets
in the rehydrated and cleaned dataset.

Expected input:

    data/tweets_cleaned.parquet

This file must contain at least:
    id
    user_id
    created_at
    action
    action_id

The repository itself only provides tweet IDs and enrichment variables.
Users must first rehydrate the tweets and reconstruct retweet information.
"""

import pandas as pd

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"

PERIOD_1_PATH = PROJECT_DIR / "results" / "period_1"
PERIOD_2_PATH = PROJECT_DIR / "results" / "period_2"
ALL_PERIODS_PATH = PROJECT_DIR / "results" / "all_periods"

PERIOD_1_PATH.mkdir(parents=True, exist_ok=True)
PERIOD_2_PATH.mkdir(parents=True, exist_ok=True)
ALL_PERIODS_PATH.mkdir(parents=True, exist_ok=True)


# Define Periods
start_date_period_1 = "2021-10-13"
end_date_period_1 = "2022-02-23"

start_date_period_2 = "2022-02-24"
end_date_period_2 = "2022-07-22"

start_date_all_periods = "2021-10-13"
end_date_all_periods = "2022-07-22"


# %%
# Load data
df = pd.read_parquet(DATA_PATH / "tweets_cleaned.parquet")

required_columns = ["id", "user_id", "created_at", "action", "action_id"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}. "
        "Please make sure tweets_cleaned.parquet was produced by the previous scripts "
        "and contains retweet information from the rehydrated tweets."
    )

df["created_at"] = pd.to_datetime(df["created_at"])
df["id"] = df["id"].astype(str)
df["action_id"] = df["action_id"].astype(str)

# Use all tweets for network analysis
df_network = df.copy()


# %%
# Split tweets according to the periods
period_1 = df_network[
    (df_network["created_at"] >= start_date_period_1) &
    (df_network["created_at"] <= end_date_period_1)
]

period_2 = df_network[
    (df_network["created_at"] >= start_date_period_2) &
    (df_network["created_at"] <= end_date_period_2)
]

all_periods = df_network[
    (df_network["created_at"] >= start_date_all_periods) &
    (df_network["created_at"] <= end_date_all_periods)
]


# %%
# Keep retweets ONLY and create nodes and edges

# Period 1
retweets_period_1 = period_1[period_1["action"] == "retweeted"].copy()

# Period 2
retweets_period_2 = period_2[period_2["action"] == "retweeted"].copy()

# All periods
retweets_all_periods = all_periods[all_periods["action"] == "retweeted"].copy()


# Convert retweet IDs to str
# Period 1
retweets_period_1["action_id"] = retweets_period_1["action_id"].astype(str)

# Period 2
retweets_period_2["action_id"] = retweets_period_2["action_id"].astype(str)

# All periods
retweets_all_periods["action_id"] = retweets_all_periods["action_id"].astype(str)


# Merge the two data frames, in order to find all retweets among MPs
# Period 1
retweets_period_1_merged = pd.merge(
    retweets_period_1,
    period_1,
    left_on="action_id",
    right_on="id",
    suffixes=("_retweet", "_original")
)

# Period 2
retweets_period_2_merged = pd.merge(
    retweets_period_2,
    period_2,
    left_on="action_id",
    right_on="id",
    suffixes=("_retweet", "_original")
)

# All periods
retweets_all_periods_merged = pd.merge(
    retweets_all_periods,
    all_periods,
    left_on="action_id",
    right_on="id",
    suffixes=("_retweet", "_original")
)


# Create edges
# Period 1
edges_period_1 = retweets_period_1_merged[["user_id_retweet", "user_id_original"]]
edges_period_1 = edges_period_1.rename(columns={"user_id_retweet": "source", "user_id_original": "target"})

# Remove edges where users retweet themselves
edges_period_1 = edges_period_1[edges_period_1["source"] != edges_period_1["target"]]


# Period 2
edges_period_2 = retweets_period_2_merged[["user_id_retweet", "user_id_original"]]
edges_period_2 = edges_period_2.rename(columns={"user_id_retweet": "source", "user_id_original": "target"})

# Remove edges where users retweet themselves
edges_period_2 = edges_period_2[edges_period_2["source"] != edges_period_2["target"]]


# All periods
edges_all_periods = retweets_all_periods_merged[["user_id_retweet", "user_id_original"]]
edges_all_periods = edges_all_periods.rename(columns={"user_id_retweet": "source", "user_id_original": "target"})

# Remove edges where users retweet themselves
edges_all_periods = edges_all_periods[edges_all_periods["source"] != edges_all_periods["target"]]


# Create nodes
# Period 1
unique_nodes_period_1 = pd.unique(edges_period_1[["source", "target"]].values.ravel("K"))

nodes_period_1 = pd.DataFrame({"label": unique_nodes_period_1})
nodes_period_1.insert(0, "id", range(1, len(nodes_period_1) + 1))  # Add 'id' column


# Period 2
unique_nodes_period_2 = pd.unique(edges_period_2[["source", "target"]].values.ravel("K"))

nodes_period_2 = pd.DataFrame({"label": unique_nodes_period_2})
nodes_period_2.insert(0, "id", range(1, len(nodes_period_2) + 1))  # Add 'id' column


# All periods
unique_nodes_all_periods = pd.unique(edges_all_periods[["source", "target"]].values.ravel("K"))

nodes_all_periods = pd.DataFrame({"label": unique_nodes_all_periods})
nodes_all_periods.insert(0, "id", range(1, len(nodes_all_periods) + 1))  # Add 'id' column


# %%
# Save edges and nodes

# Period 1
edges_period_1.to_csv(PERIOD_1_PATH / "edges.csv", index=False)
nodes_period_1.to_csv(PERIOD_1_PATH / "nodes.csv", index=False)

# Period 2
edges_period_2.to_csv(PERIOD_2_PATH / "edges.csv", index=False)
nodes_period_2.to_csv(PERIOD_2_PATH / "nodes.csv", index=False)

# All periods
edges_all_periods.to_csv(ALL_PERIODS_PATH / "edges.csv", index=False)
nodes_all_periods.to_csv(ALL_PERIODS_PATH / "nodes.csv", index=False)

# %%
