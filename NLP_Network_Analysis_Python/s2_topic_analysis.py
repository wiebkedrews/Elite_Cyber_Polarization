# %%
#####################
###### BERTOPIC #####
#####################

# From here: https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html#preventing-stochastic-behavior
# BERTopic. NOTE: We include all tweets and retweets for topic analysis

"""
Replication note
----------------
This script requires the cleaned tweet dataset produced by:

    s1_tweet_cleaner.py

Expected input:

    data/tweets_cleaned.parquet

This file must contain at least:
    id
    created_at
    text_clean

The repository itself only provides tweet IDs and enrichment variables.
Users must first rehydrate the tweets, machine translate them, and run
s1_tweet_cleaner.py before running this BERTopic analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from pathlib import Path

from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from tqdm import tqdm

tqdm.pandas()

SENTENCE_MODEL = "all-mpnet-base-v2"  # The default is: 'all-MiniLM-L6-v2'

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 384 dimensional dense vector space
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2 768 dimensional dense vector space

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"
BERTOPIC_RESULTS_PATH = PROJECT_DIR / "results" / "bertopic_results"

BERTOPIC_RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# %%
# Load tweets (clean and translated)
df = pd.read_parquet(DATA_PATH / "tweets_cleaned.parquet")


# %%
# NOTE: We include all tweets and retweets for topic analysis

# Convert tweets to list
df_text_clean = df["text_clean"].tolist()

# Pre-calculate embeddings
embedding_model = SentenceTransformer(SENTENCE_MODEL, device="cuda")
embeddings = embedding_model.encode(df_text_clean, show_progress_bar=True)

# Preventing stochastic behavior
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)

# Controlling number of topics
hdbscan_model = HDBSCAN(min_cluster_size=500, metric="euclidean", cluster_selection_method="eom", prediction_data=True)

# Improving default representation
vectorizer_model = CountVectorizer(stop_words="english", min_df=10, ngram_range=(1, 3))

# Additional representations
representation_model = MaximalMarginalRelevance(diversity=0.3)

# All steps together
topic_model = BERTopic(

    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,

    # Hyperparameters
    top_n_words=10,
    verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(df_text_clean, embeddings)


# %%
# Save the model to a file
topic_model.save(BERTOPIC_RESULTS_PATH / "bertopic_model")


# %%
# Load the model from the file
# topic_model = BERTopic.load(BERTOPIC_RESULTS_PATH / "bertopic_model")


# %%
# Show frequent topics
pd.set_option("display.max_rows", None)
topic_model.get_topic_info()


# %%
# Reduce outliers using the `embeddings` strategy
topics = topic_model.reduce_outliers(df_text_clean, topics, strategy="embeddings", embeddings=embeddings)

topic_model.update_topics(df_text_clean, topics=topics, vectorizer_model=vectorizer_model)


# %%
# Show frequent topics
topic_info = topic_model.get_topic_info()
print(topic_info)

# Save the table to a CSV file
topic_info.to_csv(BERTOPIC_RESULTS_PATH / "topic_info.csv", index=False)


# %%
# Hierarchical Clustering
hierarchical_topics = topic_model.hierarchical_topics(df_text_clean)

fig_hc = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig_hc

# Save hc_cluster
fig_hc.write_html(BERTOPIC_RESULTS_PATH / "hierarchical_clustering_topics_before_merging.html")


# %%
# Merge topics: Russia-Ukraine with 'Gas Oil Russian Energy', 'Germany German Ukraine Russia' (about weapon delivery), 'Moldova, Ukraine, Moldova, Romania' (Ukraine EU status), 'Refugees Ukraine Ukrainian fleeing' and a topic that was attested close relationship to Russia-Ukraine in the hierarchical clustering: 'war and peace'
topics_to_merge = [0, 13, 44, 45, 47, 49]

topic_model.merge_topics(df_text_clean, topics_to_merge)


# %%
# Hierarchical Clustering
hierarchical_topics = topic_model.hierarchical_topics(df_text_clean)

fig_hc = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig_hc

# Save hc_cluster
fig_hc.write_html(BERTOPIC_RESULTS_PATH / "hierarchical_clustering_topics_after_merging.html")


# %%
# Update topics
topics = topic_model.topics_

# Show frequent topics
topic_info = topic_model.get_topic_info()
print(topic_info)

# Save the table to a CSV file
topic_info.to_csv(BERTOPIC_RESULTS_PATH / "topic_info_after_merging.csv", index=False)


# %%
# Define custom names for the top 10 topics to appear in the graph
# custom_topic_names = {
#     0: "Russia-Ukraine ('ukraine', 'russia', 'russian', 'putin')",
#     1: "Elections ('party', 'vote', 'president', 'election')",
#     2: "EP ('european', 'europe', 'eu', 'parliament')",
#     3: "Spain ('spain', 'vox', 'spanish', 'snchez',)",
#     4: "France ('macron', 'french', 'france', 'le', 'emmanuel')",
#     5: "Generic ('yes', 'know', 'like', 'just')",
#     6: "Covid-19 ('covid', 'health', 'vaccination', 'pandemic')",
#     7: "Agriculture ('food', 'farmers', 'agriculture', 'agricultural')",
#     8: "Poland ('poland', 'polish', 'pis', 'kaczyski')",
#     9: "Climate ('climate', 'emissions', 'climate change', 'carbon')",
# }


# %%
###############################################
###### Visualizations of topics over time #####
###############################################

# Visualization of all topics over time

# Ensure 'created_at' is in datetime format
df["created_at"] = pd.to_datetime(df["created_at"])

# Make unique timestamps per week per tweet
timestamps = df["created_at"].dt.to_period("W").dt.to_timestamp()

# Extract the cleaned text data from the DataFrame as a list of strings
documents = df["text_clean"].tolist()

# Generate topics over time
topics_over_time = topic_model.topics_over_time(documents, timestamps.to_list())

# Visualization of all topics over time
fig_all_topics = topic_model.visualize_topics_over_time(topics_over_time)

# Add a vertical line to the Plotly figure
line = go.layout.Shape(
    type="line",
    x0="2022-02-24", x1="2022-02-24",
    y0=0, y1=1, yref="paper",
    line=dict(
        color="Red",
        width=2,
        dash="dash",
    )
)

# Add the line to the layout
fig_all_topics.update_layout(shapes=[line])

# Optionally, add an annotation
fig_all_topics.add_annotation(
    x="2022-02-24", y=0.95, yref="paper",
    text="Russian Invasion",
    showarrow=False,
    font=dict(
        color="Red"
    )
)

# Show the figure
fig_all_topics.show()

# Save the figure
fig_all_topics.write_image(BERTOPIC_RESULTS_PATH / "all_topics_over_time.png")


# %%
# Add topic number to df and select topics
df["topic"] = topics

# Create 'russo_ukraine'
# It will be 1 if 'topic' is 0 (i.e., Russo-Ukraine), and 0 otherwise
df["russo_ukraine"] = df["topic"].apply(lambda x: 1 if x == 0 else 0)

# Save as separate data set
df.to_parquet(DATA_PATH / "tweets_with_topic.parquet", index=False)


# %%
# Select the first 1,000 observations of the variable 'text_clean'
df_top1000 = df.head(1000)

# Display the first 1,000 observations
pd.set_option("display.max_rows", None)
print(df_top1000)

# %%
