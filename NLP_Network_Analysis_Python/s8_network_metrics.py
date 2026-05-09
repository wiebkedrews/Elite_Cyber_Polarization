# %%
#########################################
### NETWORK METRICS & ROBUSTNESS #######
#########################################

"""
Replication note
----------------
This script computes additional network metrics as robustness checks
for the retweet network and ECS community structure.

Expected input:

    results/all_periods/
        nodes_gephi.csv
        edges_gephi.csv

The script computes:

- E-I index
- assortativity
- density
- clustering coefficients
- transitivity
- structural hole constraint

Parts of the E-I index implementation were adapted from:

Bruns, A., Kasianenko, K., Suresh, V. P., Dehghan, E., & Vodden, L. (2025).
"Untangling the Furball: A Practice Mapping Approach to the Analysis of
Multimodal Interactions in Social Networks."
Social Media + Society, 11(2).

https://doi.org/10.1177/20563051251331748

Parts of the assortativity workflow were inspired by:

Nasuto, A., & Rowe, F. (2024).
"Understanding anti-immigration sentiment spreading on Twitter."
PLOS ONE, 19(9), e0307917.

https://doi.org/10.1371/journal.pone.0307917
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from networkx.algorithms.structuralholes import constraint


# %%
# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Define path
# DATA_PATH = PROJECT_DIR / "results" / "period_1"
# DATA_PATH = PROJECT_DIR / "results" / "period_2"

DATA_PATH = PROJECT_DIR / "results" / "all_periods"


# %%
def load_network(
    path,
    directed=True,
    drop_self_loops=True,
    restrict_to_known_nodes=False,
):
    """
    Load node and edge tables, normalize IDs, collapse duplicate edges
    into weights, and build a NetworkX graph.
    """

    nodes = pd.read_csv(os.path.join(path, "nodes_gephi.csv"))
    edges = pd.read_csv(os.path.join(path, "edges_gephi.csv"))

    required_node_cols = {"id"}
    required_edge_cols = {"source", "target"}

    missing_node_cols = required_node_cols - set(nodes.columns)
    missing_edge_cols = required_edge_cols - set(edges.columns)

    if missing_node_cols:
        raise ValueError(
            f"Missing required columns in nodes_gephi.csv: "
            f"{sorted(missing_node_cols)}"
        )

    if missing_edge_cols:
        raise ValueError(
            f"Missing required columns in edges_gephi.csv: "
            f"{sorted(missing_edge_cols)}"
        )

    nodes = nodes.copy()
    edges = edges.copy()

    nodes["id"] = nodes["id"].astype(str)
    edges[["source", "target"]] = edges[["source", "target"]].astype(str)

    print("Nodes columns:", nodes.columns.tolist())
    print("Edges columns:", edges.columns.tolist())

    edges = edges.dropna(subset=["source", "target"]).copy()

    if drop_self_loops:
        edges = edges.loc[
            edges["source"] != edges["target"]
        ].copy()

    if restrict_to_known_nodes:
        valid_ids = set(nodes["id"])

        n_before = len(edges)

        edges = edges[
            edges["source"].isin(valid_ids) &
            edges["target"].isin(valid_ids)
        ].copy()

        print(
            f"Restricted edges to known nodes: "
            f"kept {len(edges):,} of {n_before:,} rows"
        )

    # Collapse duplicate edges into weights
    edges_weighted = (
        edges.groupby(["source", "target"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )

    # Build graph
    G = nx.DiGraph() if directed else nx.Graph()

    print(
        f"Creating {'directed' if directed else 'undirected'} "
        f"graph ({type(G).__name__})"
    )

    G.add_nodes_from(nodes["id"])

    G.add_weighted_edges_from(
        edges_weighted[
            ["source", "target", "weight"]
        ].itertuples(index=False, name=None)
    )

    # Add node attributes
    node_attrs = nodes.set_index("id").to_dict("index")
    nx.set_node_attributes(G, node_attrs)

    extra_graph_nodes = sorted(
        set(G.nodes()) - set(nodes["id"])
    )

    if extra_graph_nodes:
        print(
            f"Warning: {len(extra_graph_nodes)} graph nodes appear "
            f"in edges but not in nodes_gephi.csv"
        )

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    return G, nodes, edges_weighted


# %%
def build_node_lookup(nodes, community_col="community", label_col="label"):
    """
    Create normalized lookup structures used by downstream functions.
    """

    nodes_norm = nodes.copy()

    required_cols = ["id", community_col]

    missing_cols = [
        c for c in required_cols
        if c not in nodes_norm.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in nodes table: {missing_cols}"
        )

    nodes_norm["id"] = nodes_norm["id"].astype(str)

    node_to_comm = dict(
        zip(nodes_norm["id"], nodes_norm[community_col])
    )

    node_info_cols = ["id", community_col]

    if label_col in nodes_norm.columns:
        node_info_cols.append(label_col)

    node_info = nodes_norm[node_info_cols].copy()

    return nodes_norm, node_to_comm, node_info


# %%
def ei_index(G, node_to_comm, community_values=None, use_weights=True):
    """
    Compute E-I index for each community.
    """

    results = []

    if community_values is None:
        community_values = sorted(set(node_to_comm.values()))

    internal_total = 0
    external_total = 0

    for community in community_values:

        internal = 0
        external = 0

        community_nodes = [
            n for n, c in node_to_comm.items()
            if c == community
        ]

        for node in community_nodes:

            for neighbor in G.neighbors(node):

                weight = (
                    G[node][neighbor].get("weight", 1)
                    if use_weights
                    else 1
                )

                if node_to_comm.get(neighbor) == community:
                    internal += weight
                else:
                    external += weight

        denominator = internal + external

        ei = (
            (external - internal) / denominator
            if denominator > 0
            else np.nan
        )

        results.append({
            "community": community,
            "internal": internal,
            "external": external,
            "ei_index": ei,
        })

        internal_total += internal
        external_total += external

    overall_denominator = internal_total + external_total

    overall_ei = (
        (external_total - internal_total) / overall_denominator
        if overall_denominator > 0
        else np.nan
    )

    return pd.DataFrame(results), overall_ei


# %%
def compute_network_metrics(G, community_attr="community"):
    """
    Compute global network metrics.
    """

    metrics = {}

    metrics["density"] = nx.density(G)

    try:
        metrics["transitivity"] = nx.transitivity(G.to_undirected())
    except Exception:
        metrics["transitivity"] = np.nan

    try:
        metrics["average_clustering_unweighted"] = nx.average_clustering(
            G.to_undirected()
        )
    except Exception:
        metrics["average_clustering_unweighted"] = np.nan

    try:
        metrics["average_clustering_weighted"] = nx.average_clustering(
            G.to_undirected(),
            weight="weight"
        )
    except Exception:
        metrics["average_clustering_weighted"] = np.nan

    try:
        metrics["assortativity"] = nx.attribute_assortativity_coefficient(
            G,
            community_attr
        )
    except Exception:
        metrics["assortativity"] = np.nan

    return metrics


# %%
# STEP 1: LOAD NETWORK

print("\n" + "=" * 70)
print("STEP 1. LOAD RETWEET NETWORK")
print("=" * 70)

G_p, nodes_p, edges_weighted_p = load_network(
    path=DATA_PATH,
    directed=True,
    drop_self_loops=True,
    restrict_to_known_nodes=True,
)

print("\nNetwork summary")
print("-" * 70)
print(f"Nodes: {G_p.number_of_nodes()}")
print(f"Edges: {G_p.number_of_edges()}")
print("Note: edges represent retweet actions")


# %%
# STEP 2: BUILD LOOKUPS

nodes_p_norm, node_to_comm_p, node_info_p = build_node_lookup(
    nodes_p,
    community_col="community",
    label_col="label",
)

community_values_p = sorted(
    nodes_p_norm["community"].dropna().unique()
)

print("\nDetected communities")
print("-" * 70)
print(community_values_p)


# %%
# STEP 3: E-I INDEX

ei_unw_p, ei_overall_unw_p = ei_index(
    G=G_p,
    node_to_comm=node_to_comm_p,
    community_values=community_values_p,
    use_weights=False,
)

print("\nUnweighted E-I index")
print("-" * 70)
print(ei_unw_p)

print("\nOverall unweighted E-I")
print("-" * 70)
print(ei_overall_unw_p)


# %%
# Weighted E-I

ei_w_p, ei_overall_w_p = ei_index(
    G=G_p,
    node_to_comm=node_to_comm_p,
    community_values=community_values_p,
    use_weights=True,
)

print("\nWeighted E-I index")
print("-" * 70)
print(ei_w_p)

print("\nOverall weighted E-I")
print("-" * 70)
print(ei_overall_w_p)


# %%
# STEP 4: GLOBAL NETWORK METRICS

metrics_p = compute_network_metrics(
    G_p,
    community_attr="community"
)

print("\nAssortativity by community")
print("-" * 70)
print(metrics_p["assortativity"])

print("\nGraph density")
print("-" * 70)
print(metrics_p["density"])

print("\nAverage clustering")
print("-" * 70)
print(f"Unweighted: {metrics_p['average_clustering_unweighted']}")
print(f"Weighted:   {metrics_p['average_clustering_weighted']}")

print("\nGlobal clustering coefficient (transitivity)")
print("-" * 70)
print(metrics_p["transitivity"])


# %%
# STEP 5: STRUCTURAL HOLES / CONSTRAINT

print("\nStructural hole constraint")
print("-" * 70)

constraint_scores = constraint(G_p)

constraint_df = pd.DataFrame({
    "node": list(constraint_scores.keys()),
    "constraint": list(constraint_scores.values())
})

print(constraint_df.head())

# %%
