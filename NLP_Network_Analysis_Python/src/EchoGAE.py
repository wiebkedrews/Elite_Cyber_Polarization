"""
Graph Autoencoder (GAE) implementation used for Echo Chamber Score analysis.

This implementation is part of the ECS pipeline building on:

"Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach"

by Faisal Alatawi, Paras Sheth, and Huan Liu.

GitHub repository:
https://github.com/faalatawi/echo-chamber-score
"""

import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

from .GAE import run


def EchoGAE_algorithm(
    G,
    user_embeddings=None,
    show_progress=True,
    epochs=300,
    hidden_channels=100,
    out_channels=50,
) -> np.ndarray:

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create node features
    if user_embeddings is None:
        X = torch.eye(len(G.nodes), dtype=torch.float32, device=DEVICE)

    else:
        X = []

        for node in G.nodes:
            X.append(user_embeddings[node])

        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    # Create edge list
    edge_list = np.array(G.edges).T
    edge_list = torch.tensor(edge_list, dtype=torch.int64).to(DEVICE)

    # Create PyTorch Geometric data object
    data = Data(x=X, edge_index=edge_list)
    data = train_test_split_edges(data)

    # Run the model
    model, x, train_pos_edge_index = run(
        data,
        show_progress=show_progress,
        epochs=epochs,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    # Generate embeddings
    GAE_embedding = model.encode(
        x,
        train_pos_edge_index
    ).detach().cpu().numpy()

    return GAE_embedding
