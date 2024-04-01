import numpy as np
import pandas as pd
import logging


logger = logging.getLogger()


def knn():
    k = 10

    data = "../preprocessing/spatial/data/distance_matrix.csv"

    df = pd.read_csv(data, index_col=0)

    df.replace(0, float("inf"), inplace=True)

    distance_dict = {}
    for i in range(len(df)):
        distance_dict[df.index.values[i]] = df.values[i].tolist()

    index_sorted_distance_dict = {}
    for number, distances in distance_dict.items():
        sorted_distances = sorted(distances)
        mapped_distance_list = [distances.index(x) + 1 for x in sorted_distances][:k]
        index_sorted_distance_dict[number] = mapped_distance_list

    spatial_adj = np.zeros((len(df.columns), len(df.index)))

    for number, distances in index_sorted_distance_dict.items():
        for dist in distances:
            spatial_adj[number - 1, dist - 1] = 1

    return spatial_adj


if __name__ == '__main__':
    knn()
