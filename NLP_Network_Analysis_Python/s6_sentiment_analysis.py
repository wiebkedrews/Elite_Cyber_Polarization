# %%
########################
### SENTIMENT ANALYSIS #
########################

"""
Replication note
----------------
This script runs VADER sentiment analysis on cleaned translated tweets.

Expected input:

    data/tweets_cleaned_community.parquet

This file is produced after running the text-cleaning, network,
embedding, and ECS/community scripts.

The script requires at least:
    text_clean
    text_translated
    created_at

Parts of the VADER sentiment workflow and visualization structure were
inspired by publicly available tutorials and notebooks, including:

GeeksforGeeks:
https://www.geeksforgeeks.org/nlp/twitter-sentiment-analysis-on-russia-ukraine-war-using-python/

Kaggle notebook:
https://www.kaggle.com/code/scratchpad/notebook0a29e8a9a2/edit

The implementation was substantially adapted and extended for the present
analysis pipeline.
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# %%
# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data"
SENTIMENT_RESULTS_PATH = PROJECT_DIR / "results" / "sentiment_results"

SENTIMENT_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PATH / "tweets_cleaned_community.parquet"
OUTPUT_FILE = DATA_PATH / "tweets_cleaned_community_sentiment.parquet"
OUTPUT_CSV = DATA_PATH / "tweets_cleaned_community_sentiment.csv"


# %%
# Download VADER lexicon if necessary
nltk.download("vader_lexicon", quiet=True)

analyzer = SentimentIntensityAnalyzer()


# %%
# Load data
tweets_df = pd.read_parquet(INPUT_FILE)

required_columns = ["text_clean", "text_translated", "created_at"]
missing_columns = [col for col in required_columns if col not in tweets_df.columns]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}. "
        "Please make sure the previous scripts were run successfully."
    )


# %%
# Run sentiment analysis
scores = []

for text in tweets_df["text_clean"]:
    score = analyzer.polarity_scores(str(text))
    scores.append({
        "sentiment_compound": score["compound"],
        "positive": score["pos"],
        "negative": score["neg"],
        "neutral": score["neu"],
    })

sentiments_score = pd.DataFrame.from_dict(scores)
tweets_df = tweets_df.join(sentiments_score)


# %%
# Create a variable that identifies extreme sentiments

conditions_1 = [
    (tweets_df["sentiment_compound"] <= -0.6),
    (tweets_df["sentiment_compound"] > -0.6) & (tweets_df["sentiment_compound"] <= -0.2),
    (tweets_df["sentiment_compound"] > -0.2) & (tweets_df["sentiment_compound"] <= 0.2),
    (tweets_df["sentiment_compound"] > 0.2) & (tweets_df["sentiment_compound"] <= 0.6),
    (tweets_df["sentiment_compound"] > 0.6),
]

values = [
    "extremely negative",
    "negative",
    "neutral",
    "positive",
    "extremely positive",
]

tweets_df["extreme_sentiment"] = np.select(conditions_1, values)


# %%
# Pie chart of extreme sentiments

df_plot = (
    pd.DataFrame(
        tweets_df.groupby(["extreme_sentiment"])["extreme_sentiment"].count()
    )
    .rename(columns={"extreme_sentiment": "Counts"})
    .assign(Percentage=lambda x: (x.Counts / x.Counts.sum()) * 100)
)

order = [
    "extremely negative",
    "negative",
    "neutral",
    "positive",
    "extremely positive",
]

df_plot = df_plot.reindex(order)

colors = {
    "extremely negative": "darkred",
    "negative": "lightcoral",
    "neutral": "white",
    "positive": "lightblue",
    "extremely positive": "darkblue",
}

pie_colors = [colors[sentiment] for sentiment in order]
labels = [f"{index} [{int(row['Counts'])}]" for index, row in df_plot.iterrows()]

fig, axe = plt.subplots(figsize=(10, 5))
wedges, texts, autotexts = axe.pie(
    df_plot["Counts"].values,
    startangle=90,
    autopct="%2.2f%%",
    colors=pie_colors,
)

for wedge in wedges:
    wedge.set_edgecolor("black")

for autotext in autotexts:
    autotext.set(size=8)

plt.style.use("default")
axe.legend(
    wedges,
    labels,
    title="Sentiments",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    prop={"size": 8},
)

plt.title("% of Sentiment Tweets with Extreme Values")
plt.axis("equal")

fig = px.pie(
    df_plot,
    values="Counts",
    names=df_plot.index,
    color=df_plot.index,
    color_discrete_map=colors,
    title="% of Sentiment Tweets with Extreme Values",
)

fig.write_html(SENTIMENT_RESULTS_PATH / "extreme_sentiment_pie_chart.html")

plt.show()


# %%
# Create a variable that identifies sentiments in three categories

conditions_2 = [
    (tweets_df["sentiment_compound"] <= -0.4),
    (tweets_df["sentiment_compound"] > -0.4) & (tweets_df["sentiment_compound"] <= 0.4),
    (tweets_df["sentiment_compound"] > 0.4),
]

values = ["negative", "neutral", "positive"]

tweets_df["sentiment"] = np.select(conditions_2, values)


# %%
# Pie chart of three sentiment categories

df_plot = (
    pd.DataFrame(
        tweets_df.groupby(["sentiment"])["sentiment"].count()
    )
    .rename(columns={"sentiment": "Counts"})
    .assign(Percentage=lambda x: (x.Counts / x.Counts.sum()) * 100)
)

order = ["negative", "neutral", "positive"]

df_plot = df_plot.reindex(order)

colors = {
    "negative": "darkred",
    "neutral": "white",
    "positive": "darkblue",
}

pie_colors = [colors[sentiment] for sentiment in order]
labels = [f"{index} [{int(row['Counts'])}]" for index, row in df_plot.iterrows()]

fig, axe = plt.subplots(figsize=(10, 5))
wedges, texts, autotexts = axe.pie(
    df_plot["Counts"].values,
    startangle=90,
    autopct="%2.2f%%",
    colors=pie_colors,
)

for wedge in wedges:
    wedge.set_edgecolor("black")

for autotext in autotexts:
    autotext.set(size=8)

plt.style.use("default")
axe.legend(
    wedges,
    labels,
    title="Sentiments",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    prop={"size": 8},
)

plt.title("% of Sentiment Tweets")
plt.axis("equal")

fig = px.pie(
    df_plot,
    values="Counts",
    names=df_plot.index,
    color=df_plot.index,
    color_discrete_map=colors,
    title="% of Sentiment",
)

fig.write_html(SENTIMENT_RESULTS_PATH / "sentiment_pie_chart.html")

plt.show()


# %%
# Top 50 positive tweets
tweets_df.nlargest(n=50, columns=["sentiment_compound"])["text_translated"]


# %%
# Top 50 negative tweets
tweets_df.nsmallest(n=50, columns=["sentiment_compound"])["text_translated"]


# %%
# Sentiment over time
# Change of absolute sentiments over time per week

tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"])

weekly_counts = (
    tweets_df
    .groupby([pd.Grouper(key="created_at", freq="W"), "extreme_sentiment"])
    .size()
    .reset_index(name="count")
)

last_week_with_data = weekly_counts["created_at"].max()
weekly_counts = weekly_counts[weekly_counts["created_at"] <= last_week_with_data]

color_map = {
    "extremely positive": "darkblue",
    "positive": "lightblue",
    "neutral": "white",
    "negative": "lightcoral",
    "extremely negative": "darkred",
}

fig = px.bar(
    weekly_counts,
    x="created_at",
    y="count",
    color="extreme_sentiment",
    title="Weekly Absolute Number of Sentiment Tweets",
    labels={"created_at": "Week", "count": "Number of Tweets"},
    category_orders={
        "extreme_sentiment": [
            "extremely positive",
            "positive",
            "neutral",
            "negative",
            "extremely negative",
        ]
    },
    color_discrete_map=color_map,
)

invasion_date = "2022-02-24"

fig.add_vline(
    x=invasion_date,
    line_width=2,
    line_dash="dash",
    line_color="orange",
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="orange", width=2, dash="dash"),
        name="Russian Invasion",
    )
)

fig.update_layout(legend_title_text="Legend")

fig.write_html(SENTIMENT_RESULTS_PATH / "absolute_sentiment_over_time.html")

fig.show()


# %%
# Change of relative sentiments over time per week

total_tweets_per_week = (
    tweets_df
    .groupby(pd.Grouper(key="created_at", freq="W"))
    .size()
    .reset_index(name="total_count")
)

weekly_counts = weekly_counts.merge(total_tweets_per_week, on="created_at")
weekly_counts["proportion"] = weekly_counts["count"] / weekly_counts["total_count"]

fig = px.bar(
    weekly_counts,
    x="created_at",
    y="proportion",
    color="extreme_sentiment",
    title="Weekly Relative Sentiment Tweets",
    labels={"created_at": "Week", "proportion": "Proportion of Tweets"},
    category_orders={
        "extreme_sentiment": [
            "extremely positive",
            "positive",
            "neutral",
            "negative",
            "extremely negative",
        ]
    },
    color_discrete_map=color_map,
)

fig.update_layout(barmode="stack")

fig.add_vline(
    x=invasion_date,
    line_width=2,
    line_dash="dash",
    line_color="orange",
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="orange", width=2, dash="dash"),
        name="Russian Invasion",
    )
)

fig.update_layout(legend_title_text="Legend")

fig.write_html(SENTIMENT_RESULTS_PATH / "relative_sentiment_over_time.html")

fig.show()


# %%
# Save the DataFrame

tweets_df.to_parquet(OUTPUT_FILE, index=False)

# Optional CSV export for readability in other programs
tweets_df.to_csv(OUTPUT_CSV, index=False)

# %%
