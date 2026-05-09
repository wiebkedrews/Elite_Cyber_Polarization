# %%
#############################
### USER-LEVEL EMBEDDINGS ###
#############################

"""
Replication note
----------------
This script prepares user-level tweet embeddings for ideology detection.

Expected input:

    data/tweets_cleaned.parquet

This file must contain at least:
    user_id
    created_at
    text_translated
    action

The script uses all original tweets, excluding retweets.
"""

import pandas as pd
import numpy as np

from pathlib import Path

from sentence_transformers import SentenceTransformer
from src.tweet_preprocessing import preprocess_tweet_for_bert

from tqdm import tqdm

tqdm.pandas()


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"

PERIOD_1_PATH = PROJECT_DIR / "results" / "period_1"
PERIOD_2_PATH = PROJECT_DIR / "results" / "period_2"
ALL_PERIODS_PATH = PROJECT_DIR / "results" / "all_periods"

PERIOD_1_PATH.mkdir(parents=True, exist_ok=True)
PERIOD_2_PATH.mkdir(parents=True, exist_ok=True)
ALL_PERIODS_PATH.mkdir(parents=True, exist_ok=True)

SENTENCE_MODEL = "all-mpnet-base-v2"  # The default is: 'all-MiniLM-L6-v2'


# %%
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

required_columns = ["user_id", "created_at", "text_translated", "action"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}. "
        "Please make sure tweets_cleaned.parquet was produced by s1_tweet_cleaner.py."
    )

df["created_at"] = pd.to_datetime(df["created_at"])

# Rename user_id to author_id for the rest of the analysis
df = df.rename(columns={"user_id": "author_id"})

# NOTE: Keep original tweets ONLY (not retweets)
df = df[~df["action"].isin(["retweeted"])]

# Keep certain columns
df = df[["author_id", "text_translated", "created_at"]]

# Rename columns
df = df.rename(columns={"text_translated": "tweets"})

# Convert tweets to str
df["tweets"] = df["tweets"].astype(str)


# %%
# Split tweets according to the periods
# NOTE: Ideology detection and embeddings computation are done on all original tweets
# for the respective period: before, after, and all periods.

period_1 = df[
    (df["created_at"] >= start_date_period_1) &
    (df["created_at"] <= end_date_period_1)
]

period_2 = df[
    (df["created_at"] >= start_date_period_2) &
    (df["created_at"] <= end_date_period_2)
]

all_periods = df[
    (df["created_at"] >= start_date_all_periods) &
    (df["created_at"] <= end_date_all_periods)
]


# %%
# Group by author_id and convert to array and truncate tweets to a maximum of 200 per author_id

def process_period(df):
    df_grouped = df.groupby("author_id")["tweets"].apply(list).reset_index()
    df_grouped["tweets"] = df_grouped["tweets"].apply(lambda x: np.array(x[:200]))
    return df_grouped


period_1_grouped = process_period(period_1)
period_2_grouped = process_period(period_2)
all_periods_grouped = process_period(all_periods)


# %%
# Save tweets for ideology detection to all sub-folders
period_1_grouped.to_parquet(PERIOD_1_PATH / "tweets.parquet", index=False)
period_2_grouped.to_parquet(PERIOD_2_PATH / "tweets.parquet", index=False)
all_periods_grouped.to_parquet(ALL_PERIODS_PATH / "tweets.parquet", index=False)


# %%
# Compute embeddings for each period

def compute_embeddings(df_grouped, path):
    def preprocess_tweets(tweets):
        out = []
        for tw in tweets:
            tw = preprocess_tweet_for_bert(tw)
            if len(tw) > 1:
                out.append(" ".join(tw))
        return out

    df_grouped["tweets"] = df_grouped["tweets"].progress_apply(preprocess_tweets)

    # Remove users with no tweets
    df_grouped = df_grouped[df_grouped["tweets"].apply(len) > 0]

    model = SentenceTransformer(SENTENCE_MODEL)

    def embed_user_tweets(tweets):
        emb = model.encode(tweets)
        emb = np.mean(emb, axis=0)
        return emb

    df_grouped["embeddings"] = df_grouped["tweets"].progress_apply(embed_user_tweets)

    embeddings_tmp = df_grouped[["author_id", "embeddings"]]
    embeddings_tmp.reset_index(drop=True, inplace=True)

    # Save embeddings to the specified path
    embeddings_tmp.to_parquet(path / "embeddings.parquet", index=False)


# Compute and save embeddings for each period
compute_embeddings(period_1_grouped, PERIOD_1_PATH)
compute_embeddings(period_2_grouped, PERIOD_2_PATH)
compute_embeddings(all_periods_grouped, ALL_PERIODS_PATH)


# %%
