import dask
import dask.dataframe
from dask.distributed import Client
import numpy as np
import os
import pyarrow as pa
from PIL import Image
import skimage.io as imageio
from transformers import AutoImageProcessor, AutoModel
import torch
from skimage.feature import hog, SIFT
import cv2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features_random(image_paths):
    return np.random.rand(len(image_paths), 512).tolist()


def luminance(img):
    if len(img.shape) > 2:
        img = 0.2126 * img[:, :, 0] + 0.7152 * \
            img[:, :, 1] + 0.0722 * img[:, :, 2]
        img = img.astype(np.uint8)

    return img


def extract_sift(image):
    sift = SIFT()
    try:
        sift.detect_and_extract(image)
    except Exception as e:
        print(f"Error extracting SIFT features: {e}")
        return np.zeros((128,))

    if sift.descriptors is None:
        return np.zeros((128,))

    return sift.descriptors.sum(axis=0) / sift.descriptors.shape[0]


def color_features(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    mean = cv2.mean(image)[:3]
    stddev = cv2.meanStdDev(image)[1].flatten()[:3]

    return np.concatenate((hist, mean, stddev))


def extract_features_classic(image_paths): 
    features = []
    for path in image_paths:
        if not is_available(path):
            continue
        i = imageio.imread(path)

        if len(i.shape) == 4:
            i = i[:, :, :3, 0]
        if len(i.shape) == 3:
            i = i[:, :, :3]


        v = np.concatenate([
            hog(luminance(i), orientations=2, pixels_per_cell=(32, 32),
                cells_per_block=(1, 1), visualize=False, feature_vector=True),
            color_features(i)
        ])

        v = (v - np.mean(v, axis=0)) / (np.std(v, axis=0)+1e-6)

        features.append(v.tolist())


    return features

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
    client = Client(n_workers=3, threads_per_worker=4, memory_limit="2.5GB")

    df = dask.dataframe.read_csv("./data/image-files.csv")
    df = df.repartition(npartitions=1500)

    df["vector"] = df.map_partitions(
        lambda s: extract_features_classic(s["image_path"]),
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
        "./data/vectorss.parquet",
        compression="snappy",
        schema=schema,
        engine="pyarrow",
        compute=True,
        write_index=False,
    )
