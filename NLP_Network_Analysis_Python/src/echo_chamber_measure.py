"""
Echo Chamber Score measurement.

This implementation is part of the ECS pipeline building on:

"Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach"

by Faisal Alatawi, Paras Sheth, and Huan Liu.

GitHub repository:
https://github.com/faalatawi/echo-chamber-score
"""

import numpy as np

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


class EchoChamberMeasure:
    def __init__(
        self,
        users_representations: np.ndarray,
        labels: np.ndarray,
        metric: str = "euclidean",
    ):
        if metric == "euclidean":
            self.distances = euclidean_distances(users_representations)
        elif metric == "cosine":
            self.distances = cosine_distances(users_representations)
        else:
            raise ValueError("metric must be either 'euclidean' or 'cosine'")

        self.labels = labels

    def cohesion_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        node_distances = self.distances[idx, self.labels == node_label]

        return np.mean(node_distances)

    def separation_node(self, idx: int) -> float:
        node_label = self.labels[idx]

        dist = []
        for label in np.unique(self.labels):
            if label == node_label:
                continue

            dist.append(np.mean(self.distances[idx, self.labels == label]))

        return np.min(dist)

    def metric(self, idx: int) -> float:
        a = self.cohesion_node(idx)
        b = self.separation_node(idx)

        return (-a + b + max(a, b)) / (2 * max(a, b))

    def echo_chamber_index(self) -> float:
        nodes_metric = []

        for i in range(self.distances.shape[0]):
            nodes_metric.append(self.metric(i))

        return np.mean(nodes_metric)

    def community_echo_chamber_index(self, community_label: int) -> float:
        com_eci = []

        for i in range(self.distances.shape[0]):
            if self.labels[i] == community_label:
                com_eci.append(self.metric(i))

        return np.mean(com_eci)
