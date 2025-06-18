import cupy as cp

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.dataframe
import numpy as np
from cuml.cluster import KMeans, HDBSCAN


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def clusterize(vectors, min_cluster_size=5):
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdbscan.fit_predict(vectors)
    return labels


def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    df = dask.dataframe.read_parquet("data/vectors.parquet")
    df.repartition(npartitions=100)

    df = df[df["vector"].apply(is_numpy_array, meta=("vector", "bool"))].compute()

    vectors = cp.stack([cp.asarray(x) for x in df["vector"]])

    labels = clusterize(vectors)
    df["cluster"] = labels.get()

    df.to_parquet(
        "data/clusterized.parquet",
        compression="snappy",
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
