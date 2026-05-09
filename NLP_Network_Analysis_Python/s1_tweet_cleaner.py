# %%
########################
### TEXT CLEANING ######
########################

"""
Replication note
----------------
The repository only contains tweet IDs and enrichment variables in accordance
with X/Twitter Terms of Service.

Before running this script, users must:

1. Rehydrate the tweets using the shared tweet IDs.
2. Create a file called:
       data/tweets_rehydrated.parquet
3. Ensure that this file contains the columns:
       created_at
       text_translated

For machine translation, we used the Python package:
deep-translator (version 1.11.4)

GitHub:
https://github.com/nidhaloff/deep-translator

The translated text was subsequently cleaned using the procedure implemented
in this script.
"""

import contractions
import html
import numpy as np
import pandas as pd
import re
import regex

from pathlib import Path
from textacy import preprocessing


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "tweets_rehydrated.parquet"
OUTPUT_FILE = DATA_DIR / "tweets_cleaned.parquet"


# Functions
def clean_text(text):
    text = str(text)
    text = html.unescape(text)  # Replaces HTML characters
    text = re.sub('VIDEO:|AUDIO:', ' ', text)  # Removes specific tags like 'VIDEO:'
    text = re.sub(r'http\S+|www.\S+|bit.ly/\S+|pic.twitter\S+', ' ', text)  # Removes URLs
    text = preprocessing.normalize.quotation_marks(text)
    text = text.encode('ascii', 'ignore').decode()  # Removes non ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Removes unicode whitespace characters
    text = re.sub(r'\\r|\\n|\\t|\\f|\\v', ' ', text)  # For messy data, removes unicode whitespace characters
    text = re.sub(r'(^|[^@\w])@(\w{1,15})\b', ' ', text)  # Replaces twitter handles and emails
    text = preprocessing.replace.emails(text, repl=' ')
    text = re.sub(r'\$\w*', ' ', text)  # Removes tickers
    text = preprocessing.remove.accents(text)
    text = text.replace('-', ' ').replace('–', ' ')  # Replaces dashes and special characters
    text = preprocessing.normalize.unicode(text)
    text = regex.compile('[ha][ha]+ah[ha]+').sub('haha', text)
    text = text.replace('&', ' and ')
    text = re.sub(r'([A-Za-z])\1{2,}', r'\1', text)  # Removes repeated characters
    text = re.sub(
        r'\b([a-zA-Z]{1,3})(\.[a-zA-Z]{1,3}\.?)+\b',
        lambda match: match.group(0).replace('.', ''),
        text
    )  # Handles abbreviations with multiple dots
    text = contractions.fix(text, slang=True)
    text = regex.compile(
        r'(?:(?=\b(?:\p{Lu} +){2}\p{Lu})|\G(?!\A))\p{Lu}\K +(?=\p{Lu}(?!\p{L}))'
    ).sub('', text)  # Replaces kerned text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]+', ' ', text)  # Removes numbers, special characters, etc.
    text = preprocessing.normalize.whitespace(text)  # Normalises whitespace
    return text


# Load rehydrated tweets
df = pd.read_parquet(INPUT_FILE)

# Keep only required columns
df = df[
    [
        "id",
        "created_at",
        "text_translated",
        "name",
        "username",
        "day",
        "month",
        "year",
        "dob",
        "full_name",
        "sex",
        "country",
        "nat_party",
        "nat_party_abb",
        "eu_party_group",
        "eu_party_abbr",
        "commission_dummy",
        "party_id",
        "eu_position",
        "lrgen",
        "lrecon",
        "galtan",
        "eu_eu_position",
        "eu_lrgen",
        "eu_lrecon",
        "eu_galtan",
    ]
]

# Convert IDs to string
df["id"] = df["id"].astype(str)

# Clean tweets
df["text_clean"] = df["text_translated"].apply(clean_text)

# Remove empty cells
df = df.replace(r"^\s*$", np.nan, regex=True)
df = df.dropna(subset=["text_clean"])

# Save cleaned tweets
df.to_parquet(OUTPUT_FILE, index=False)

# %%
