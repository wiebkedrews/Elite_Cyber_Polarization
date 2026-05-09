"""
Prepare translated tweet data for analysis.

This script:
1. Loads raw tweet metadata from a JSON file.
2. Extracts referenced tweet information.
3. Loads translated/enriched tweet data from a Feather file.
4. Merges both datasets.
5. Cleans translated tweet text for downstream NLP analysis.
6. Saves the processed dataset as a Feather file.

Note:
Raw tweet text should not be shared publicly if this would violate X/Twitter Terms of Service.
"""

import argparse
import html
import json
import re
from pathlib import Path

import contractions
import numpy as np
import pandas as pd
import regex
from textacy import preprocessing


def clean_text(text: str) -> str:
    """Clean translated tweet text for NLP analysis."""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"VIDEO:|AUDIO:", " ", text)
    text = re.sub(r"http\S+|www\.\S+|bit\.ly/\S+|pic\.twitter\S+", " ", text)
    text = preprocessing.normalize.quotation_marks(text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\\r|\\n|\\t|\\f|\\v", " ", text)
    text = re.sub(r"(^|[^@\w])@(\w{1,15})\b", " ", text)
    text = preprocessing.replace.emails(text, repl=" ")
    text = re.sub(r"\$\w*", " ", text)
    text = preprocessing.remove.accents(text)
    text = text.replace("-", " ").replace("–", " ")
    text = preprocessing.normalize.unicode(text)
    text = regex.compile(r"[ha][ha]+ah[ha]+").sub("haha", text)
    text = text.replace("&", " and ")
    text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)

    text = re.sub(
        r"\b([a-zA-Z]{1,3})(\.[a-zA-Z]{1,3}\.?)+\b",
        lambda match: match.group(0).replace(".", ""),
        text,
    )

    text = contractions.fix(text, slang=True)
    text = regex.compile(
        r"(?:(?=\b(?:\p{Lu} +){2}\p{Lu})|\G(?!\A))\p{Lu}\K +(?=\p{Lu}(?!\p{L}))"
    ).sub("", text)

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    text = preprocessing.normalize.whitespace(text)

    return text


def extract_referenced_tweet_info(value) -> str:
    """Extract referenced tweet information from a list of dictionaries."""
    if not isinstance(value, list):
        return ""

    values = [list(d.values()) for d in value if isinstance(d, dict)]
    return ",".join(str(item) for sublist in values for item in sublist)


def prepare_dataset(raw_tweets_path: Path, translated_tweets_path: Path, output_path: Path) -> None:
    """Load, merge, clean, and save tweet data."""

    with raw_tweets_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    df_raw = pd.DataFrame(raw_data)

    raw_columns = ["id", "author_id", "created_at", "text", "referenced_tweets"]
    df_raw = df_raw[raw_columns]

    df_raw["referenced_tweets"] = df_raw["referenced_tweets"].apply(
        extract_referenced_tweet_info
    )

    split_values = df_raw["referenced_tweets"].str.split(",", n=1, expand=True)
    df_raw["action"] = split_values[0]
    df_raw["action_id"] = split_values[1]

    df_translated = pd.read_feather(translated_tweets_path)

    translated_columns = [
        "id",
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

    df_translated = df_translated[translated_columns]

    df_raw["id"] = df_raw["id"].astype(str)
    df_translated["id"] = df_translated["id"].astype(str)

    df_merged = pd.merge(df_translated, df_raw, on="id", how="inner")

    df_merged = df_merged.rename(columns={"author_id": "user_id"})

    final_columns = [
        "id",
        "user_id",
        "created_at",
        "text",
        "text_translated",
        "action",
        "action_id",
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

    df_merged = df_merged[final_columns]

    df_merged["text_clean"] = df_merged["text_translated"].apply(clean_text)

    df_merged = df_merged.replace(r"^\s*$", np.nan, regex=True)
    df_merged = df_merged.dropna(subset=["text_clean"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_feather(output_path)

    print(f"Saved processed dataset to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare translated tweet data for analysis."
    )

    parser.add_argument(
        "--raw-tweets",
        type=Path,
        required=True,
        help="Path to the raw tweet JSON file.",
    )

    parser.add_argument(
        "--translated-tweets",
        type=Path,
        required=True,
        help="Path to the translated/enriched tweet Feather file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the processed Feather file should be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepare_dataset(
        raw_tweets_path=args.raw_tweets,
        translated_tweets_path=args.translated_tweets,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
