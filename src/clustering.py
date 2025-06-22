import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dask.dataframe
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import cupy as cp
from cuml.decomposition import PCA
from cuml.cluster import HDBSCAN, KMeans
from cuml.manifold import TSNE, UMAP


def is_numpy_array(x):
    return isinstance(x, np.ndarray)


def get_xy_representation(vectors, **kwargs):
    result = TSNE(**kwargs)
    embeddings = result.fit_transform(vectors)
    return embeddings


def gen_visualization(
    vectors,
    clusters,
    prepend,
    perplexity,
    n_neighbors,
    late_exaggeration,
    learning_rate,
    metric,
    init,
):
    embeddings = get_xy_representation(
        vectors,
        n_components=2,
        random_state=42,
        learning_rate_method=None,
        init=init,
        metric=metric,
        learning_rate=learning_rate,
        late_exaggeration=late_exaggeration,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
    )

    x = embeddings[:, 0].get()
    y = embeddings[:, 1].get()

    ax = sns.scatterplot(
        x=x[clusters == -1],
        y=y[clusters == -1],
        palette="colorblind",
        hue=clusters[clusters == -1],
        s=2,
        edgecolor=None,
        linewidth=0,
    )

    ax = sns.scatterplot(
        x=x[clusters != -1],
        y=y[clusters != -1],
        palette="colorblind",
        hue=clusters[clusters != -1],
        s=2,
        edgecolor=None,
        linewidth=0,
    )

    ax.get_legend().remove()

    plt.savefig(
        f"data/imgs/plot_{prepend}_{perplexity}_{n_neighbors}_{late_exaggeration}_{learning_rate}_{metric}_{init}.png",
        dpi=300,
    )

    plt.clf()


def clusterize(vectors, min_cluster_size, min_samples):
    clusterizer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterizer.fit_predict(vectors)
    return labels


def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    df = dask.dataframe.read_parquet("data/vectors.parquet")

    df = df[df["vector"].apply(is_numpy_array, meta=("vector", "bool"))].compute()

    pca_size = int(sys.argv[1])
    min_cluster_size = int(sys.argv[2])
    min_samples = int(sys.argv[3])

    vectors = cp.stack([cp.asarray(x) for x in df["vector"]])
    vectors = (vectors - cp.mean(vectors, axis=0)) / cp.std(vectors, axis=0)

    if pca_size != -1:
        print(f"pcaing {pca_size=}")
        pca = PCA(n_components=pca_size)
        vectors = pca.fit_transform(vectors)

        df["vector"] = vectors.tolist()

    print(f"clustering {pca_size=} {min_cluster_size=} {min_samples=}")
    labels = clusterize(vectors, min_cluster_size, min_samples)

    df["cluster"] = labels.get()

    perplexities = [5, 50]
    late_exaggerations = [2.5, 1.5]
    learning_rate = 1000
    metric = "euclidean"
    init = "pca"

    for perplexity, late_exaggeration in zip(perplexities, late_exaggerations):
        n_neighbors = 3 * perplexity
        print(f"Processing {perplexity=}, {late_exaggeration=}")
        gen_visualization(
            vectors,
            df["cluster"],
            f"{pca_size if pca_size > 0 else 'orig'}_{min_cluster_size}_{min_samples}",
            perplexity,
            n_neighbors,
            late_exaggeration,
            learning_rate,
            metric,
            init,
        )

    del df["vector"]

    df.to_parquet(
        "./data/result.parquet",
        compression="snappy",
        engine="pyarrow",
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
