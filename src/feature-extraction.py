import dask
import dask.dataframe
from dask.distributed import Client
import numpy as np
import os
import pyarrow as pa
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

def extract_features_random(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None

    img = Image.open(image_path)
    img.histogram()
    return np.random.rand(512).tolist()

def extract_features_from_folder(folder_path, extensions={".jpg", ".png", ".jpeg"}):
    features = []
    image_paths = []
    for file in tqdm(os.listdir(folder_path)):
        if os.path.splitext(file)[1].lower() in extensions:
            path = os.path.join(folder_path, file)
            try:
                feat = extract_features_dinov2(path)
                features.append(feat)
                image_paths.append(file)
            except Exception as e:
                print(f"Error at {file}: {e}")
    return features, image_paths

def extract_features_dinov2(image_path):
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return None

    img = Image.open(image_path)
    img.histogram()
    print(img.height, img.width)

    inputs = processor(images=img, return_tensors="pt").to(device)
    print(inputs.pixel_values.shape)

    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_token = outputs.last_hidden_state[:, 0, :]
    print(cls_token.shape)

    feature_vector = cls_token.squeeze(0).cpu().numpy()

    return feature_vector
    

if __name__ == "__main__":

    # Image folder
    # folder = "./imgs"
    # features, paths = extract_features_from_folder(folder)
    # features_np = np.array(features)

    # print("Shape: ", features_np.shape)
    # print(features_np)

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
