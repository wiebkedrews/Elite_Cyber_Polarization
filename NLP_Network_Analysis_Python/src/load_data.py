import numpy as np
import pandas as pd

import networkx as nx
import igraph as ig
import leidenalg as la

# Disable warnings
import warnings
warnings.filterwarnings("ignore")


# Functions
def create_community_sets(user_community_dict):
    community_sets = {}

    for user_id, community in user_community_dict.items():
        if community not in community_sets:
            community_sets[community] = set()
        community_sets[community].add(user_id)

    return list(community_sets.values())


def count_tweets(x):
    return len(x)


def get_data(path: str, resolution=1, min_comm_size=0, directed_graph=True, min_degree=0, min_tweets=0):

    # 1: Get tweets & user embeddings
    df = pd.read_parquet(path + "/tweets.parquet")
    df_emb = pd.read_parquet(path + "/embeddings.parquet")

    df = df.merge(df_emb, on="author_id", how="inner")

    # Filter tweets
    df["tweet_count"] = df["tweets"].apply(count_tweets)
    df = df[df["tweet_count"] >= min_tweets]

    del df["tweet_count"]

    # 2: Graph: Read data and add weights
    df_graph_tmp = pd.read_csv(path + "/edges.csv", dtype={"source": str, "target": str})
    df_graph_tmp["weight"] = df_graph_tmp.groupby(["source", "target"]).transform("size")

    # Remove duplicates
    df_graph_tmp.drop_duplicates(subset=["source", "target", "weight"], keep="first", inplace=True)

    # Get unique nodes
    unique_nodes_tmp = set(df_graph_tmp["source"]) | set(df_graph_tmp["target"])
    print(f"\n\nNumber of nodes in RT network: {len(unique_nodes_tmp)}")

    # Filter by minimum tweets
    df_graph = df_graph_tmp[
        (df_graph_tmp["source"].isin(df["author_id"])) &
        (df_graph_tmp["target"].isin(df["author_id"]))
    ]

    unique_nodes = set(df_graph["source"]) | set(df_graph["target"])
    print(f"Number of nodes in RT network with at least {min_tweets} Tweets: {len(unique_nodes)}\n")

    if directed_graph:
        print("-- Analysing a DIRECTED graph\n")
        G = nx.from_pandas_edgelist(
            df_graph,
            "source",
            "target",
            create_using=nx.DiGraph(),
            edge_attr="weight"
        )
    else:
        print("-- Analysing an UNDIRECTED graph\n")
        G = nx.from_pandas_edgelist(
            df_graph,
            "source",
            "target",
            create_using=nx.Graph(),
            edge_attr="weight"
        )

    print(f"Number of nodes BEFORE removing self-loops: {G.number_of_nodes()}")
    print(f"Number of edges BEFORE removing self-loops: {G.number_of_edges()}")

    # Remove self-loops
    print(f"-- Removing {nx.number_of_selfloops(G)} self-loops")
    G.remove_edges_from(nx.selfloop_edges(G))

    print(f"Number of nodes AFTER removing self-loops: {G.number_of_nodes()}")
    print(f"Number of edges AFTER removing self-loops: {G.number_of_edges()}\n")

    # 3: Convert to igraph & filter the graph
    print("-- Converting networkx G to igraph g\n")
    g = ig.Graph.from_networkx(G)

    # Find LCC
    components = g.components()
    g = components.giant()

    print(f"Number of nodes in LCC: {g.vcount()}")
    print(f"Number of edges in LCC: {g.ecount()}\n")

    # Filter nodes based on degree
    filtered_nodes = [v.index for v in g.vs if g.degree(v) >= min_degree]
    g = g.subgraph(filtered_nodes)

    print(f"Number of nodes in LCC IF degree >= {min_degree}: {g.vcount()}")
    print(f"Number of edges in LCC IF degree >= {min_degree}: {g.ecount()}\n")

    # 4: Find communities using Leiden algorithm.
    # From here: https://github.com/vtraag/leidenalg/issues/157
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42
    )

    partition_dict = dict(zip(g.vs["_nx_name"], partition.membership))

    communities = create_community_sets(partition_dict)
    communities = list(communities)

    print(f"Number of communities: {len(communities)}")
    for idx, c in enumerate(communities, start=1):
        print(f"Number of members in community_{idx}: {len(c)}")

    # Remove communities if < "x" members
    communities = [s for s in communities if len(s) >= min_comm_size]

    print(f"\nNumber of FILTERED communities: {len(communities)}")
    for idx, c in enumerate(communities, start=1):
        print(f"Number of members in community_{idx}: {len(c)}")

    def which_community(node):
        for i, c in enumerate(communities):
            if node in c:
                return i
        return -1

    df["community"] = df["author_id"].apply(which_community)

    # Filter the df to keep nodes that have been assigned to a community
    df = df[df["community"] != -1]

    # Filter the graph to keep only the nodes that match the author_id from the df
    vertices_to_include = df["author_id"].tolist()
    indices_to_include = [
        i for i, v_id in enumerate(g.vs["_nx_name"])
        if v_id in vertices_to_include
    ]
    g = g.subgraph(indices_to_include)

    print(f"\nNumber of nodes FINAL (igraph): {g.vcount()}")
    print(f"Number of edges FINAL (igraph): {g.ecount()}")

    # Convert back to networkx
    print("\n-- Converting igraph g to networkx G")
    G = g.to_networkx()

    print("-- Sanity check")
    print(f"Number of nodes FINAL (networkx): {G.number_of_nodes()}")
    print(f"Number of edges FINAL (networkx): {G.number_of_edges()}\n")

    # 5: Make a map from author_id to index
    node_id_map = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, node_id_map)

    # 6: Get embeddings & labels
    users_embeddings = df.set_index("author_id")["embeddings"].to_dict()
    labels = df.set_index("author_id")["community"].to_dict()

    users_embeddings_tmp = {}
    labels_tmp = {}

    for author_id, index in node_id_map.items():
        users_embeddings_tmp[index] = users_embeddings[author_id]
        labels_tmp[index] = labels[author_id]

    users_embeddings = users_embeddings_tmp
    labels = labels_tmp
    labels = np.array(list(labels.values()))

    return G, users_embeddings, labels, node_id_map, df
