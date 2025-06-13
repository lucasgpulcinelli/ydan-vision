import dask
import dask.dataframe
from dask.distributed import Client
import numpy as np
import os
import pyarrow as pa


def extract_features_random(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None

    return np.random.rand(512)


if __name__ == "__main__":
    client = Client(n_workers=6, threads_per_worker=4, memory_limit="5GB")

    df = dask.dataframe.read_csv("./data/image-files.csv")

    df = df.repartition(npartitions=15)

    df["vector"] = df["image_path"].map_partitions(
        lambda s: s.apply(extract_features_random),
    )

    schema = pa.schema(
        [
            ("image_path", pa.string()),
            ("id", pa.string()),
            ("vector", pa.list_(pa.float32())),
        ]
    )

    df.to_parquet(
        "./data/vectors.parquet",
        compression="snappy",
        schema=schema,
        engine="pyarrow",
        compute=True,
        write_index=False,
    )
