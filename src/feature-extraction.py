import dask
import dask.dataframe
from dask.distributed import Client
import numpy as np
import os
import pyarrow as pa
from PIL import Image


def extract_features_random(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None

    img = Image.open(image_path)
    img.histogram()
    return np.random.rand(512).tolist()


if __name__ == "__main__":
    client = Client(n_workers=6, threads_per_worker=4, memory_limit="5GB")

    df = dask.dataframe.read_csv("./data/image-files.csv")

    df = df.repartition(npartitions=15)

    df["vector"] = df["image_path"].map_partitions(
        lambda s: s.apply(extract_features_random),
    )

    df = df.dropna(subset=["vector"])

    vector_length = 512
    fields = [
        pa.field("vector", pa.list_(pa.float32(), vector_length)),
        pa.field("image_path", pa.string()),
        pa.field("id", pa.string()),
    ]
    schema = pa.schema(fields)

    df.to_parquet(
        "./data/vectors.parquet",
        compression="snappy",
        engine="pyarrow",
        schema=schema,
        compute=True,
        write_index=False,
    )
