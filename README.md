# Cyber-Polarization among Political Elites in a Geopolitical Crisis

This repository contains the replication files for the article *“Cyber-Polarization among Political Elites in a Geopolitical Crisis”* by Wiebke Drews, Joris Frese, Hilke Brockmann, Pedro Fierro, Andreas Dafnos, and Daniel Triana.

---

# Data Availability

Due to X/Twitter Terms of Service, raw tweet text and associated metadata cannot be shared.

To ensure reproducibility, we provide:

- **Tweet IDs** underlying all analyses (rehydratable via the X API)
- **Enriched datasets** containing additional characteristics of Members of the European Parliament (MEPs) and European Commissioners, excluding tweet text
- Data are stored in formats suitable for GitHub:
  - `.parquet` for Python
  - split `.RData` files for R (each <25 MB)

The main replication dataset is:

- `tweet_ids_enriched.parquet`

This dataset contains:
- tweet IDs
- user IDs
- party affiliation
- country
- ideological variables
- demographic characteristics
- additional enrichment variables used throughout the analysis

---

# Rehydration and Translation

Before running the Python pipeline, users must:

1. Rehydrate the tweets using the provided tweet IDs and the X API
2. Create a file called:

```text
data/tweets_rehydrated.parquet
```

This file must contain at least:
- `id`
- `user_id`
- `created_at`
- `text_translated`
- `action`
- `action_id`

The tweet text was machine translated prior to analysis using:

- `deep-translator` (version 1.11.4)

GitHub repository:
https://github.com/nidhaloff/deep-translator

Due to X/Twitter Terms of Service, translated tweet text cannot be redistributed.

---

# Workflow Overview

The analysis proceeds in two main stages.

---

# 1. NLP & Network Analysis (Python)

Folder:

```text
NLP_Network_Analysis_Python/
```

This stage:
- cleans translated tweet text
- estimates BERTopic topic models
- constructs elite retweet networks
- computes user embeddings
- calculates Echo Chamber Scores (ECS)
- estimates sentiment
- computes additional network robustness metrics

---

## Scripts (run sequentially)

### `s1_tweet_cleaner.py`
Cleans and preprocesses translated tweets after rehydration.

### `s2_topic_analysis.py`
Runs BERTopic topic modeling and estimates topic dynamics over time.

### `s3_network_analysis.py`
Constructs retweet networks among political elites.

### `s4_ideology_detection.py`
Creates user-level embeddings for ideology and ECS analyses.

### `s5_echo_chamber_score.py`
Calculates Echo Chamber Scores (ECS) using graph autoencoders and embedding distances.

### `s6_sentiment_analysis.py`
Estimates tweet sentiment using VADER sentiment analysis.

### `s7_transform_network_for_gephi.py`
Transforms retweet network data into node and edge tables for Gephi visualization.

### `s8_network_metrics.py`
Computes additional network robustness metrics, including:
- E-I index
- assortativity
- clustering coefficients
- density
- transitivity
- structural hole constraint
  
---

## Python Outputs

The Python workflow produces:
- cleaned tweet datasets
- BERTopic topic assignments
- retweet networks
- user embeddings
- ECS scores
- sentiment scores
- community assignments
- robustness metrics

---

# 2. Statistical Analysis (R)

Folder:

```text
Statistical_Analysis_R/
```

This stage tests the study hypotheses using the outputs produced in Python.

The provided datasets already include:
- BERTopic outputs
- ECS scores
- VADER sentiment scores
- community assignments

---

## R Datasets

- split into several `.RData` files (each <25 MB)
- automatically merged in the R scripts

Two dataset generations are required:

### New dataset
Includes:
- ECS
- BERTopic
- VADER sentiment
- community variables

### Old dataset
Includes:
- engagement metrics
- legacy variables required for H3 analyses

The R scripts merge these datasets automatically.

---

## R Scripts

### `H1H2H3H4.R`

Runs:
- pairwise comparisons across ideological groups
- fixed-effects regressions
- difference-in-differences models
- robustness checks
- table and figure generation

---

## R Project File

```text
Echo Chambers.Rproj
```

---

# How to Reproduce

1. Rehydrate tweets using the provided tweet IDs and the X API
2. Machine translate tweet text
3. Create:

```text
data/tweets_rehydrated.parquet
```

4. Run Python scripts sequentially:

```text
s1 → s8
```

5. Run the R analyses:
   - open `Echo Chambers.Rproj`
   - run `H1H2H3H4.R`

The scripts:
- merge the split datasets
- estimate all models
- reproduce the article figures and tables

This workflow enables full computational reproducibility while remaining compliant with X/Twitter Terms of Service.
