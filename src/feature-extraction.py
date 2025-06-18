import dask
import dask.dataframe
from dask.distributed import Client
import numpy as np
import os
import pyarrow as pa
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features_random(image_paths):
    return np.random.rand(len(image_paths), 512).tolist()


def extract_features_dinov2(image_paths):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    imgs = []
    for path in image_paths:
        if not is_available(path):
            continue
        i = Image.open(path)
        imgs.append(i)

    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt").to(device)

        outputs = model(**inputs)

        cls_token = outputs.last_hidden_state[:, 0, :]

        feature_vector = cls_token.squeeze().cpu().numpy()

    print(f"Extracted features for {len(imgs)} images.")

    del imgs, inputs, outputs, cls_token

    return feature_vector.tolist()


def is_available(image_path):
    return isinstance(image_path, str) and os.path.exists(image_path)


if __name__ == "__main__":
    client = Client(n_workers=1, threads_per_worker=1, memory_limit="8GB")

    df = dask.dataframe.read_csv("./data/image-files.csv")
    df = df.repartition(npartitions=400)

    df["vector"] = df.map_partitions(
        lambda s: extract_features_dinov2(s["image_path"]),
        meta=("vector", "object"),
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
