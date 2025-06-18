import cupy as cp
from cuml.manifold import UMAP, TSNE

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.dataframe
import numpy as np


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def get_xy_representation(vectors):
    result = TSNE(n_components=2, perplexity=6)
    embeddings = result.fit_transform(vectors)
    return embeddings


def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    df = dask.dataframe.read_parquet("data/clusterized.parquet")
    df.repartition(npartitions=100)

    df = df[df["vector"].apply(is_numpy_array, meta=("vector", "bool"))].compute()

    vectors = cp.stack([cp.asarray(x) for x in df["vector"]])

    embeddings = get_xy_representation(vectors)

    df["x"] = embeddings[:, 0].get()
    df["y"] = embeddings[:, 1].get()

    del df["vector"]

    df.to_parquet(
        "data/result.parquet",
        compression="snappy",
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
