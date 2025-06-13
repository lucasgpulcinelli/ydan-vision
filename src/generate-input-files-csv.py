import os
import sys

import base64
import pandas as pd


def decode_base64(filename):
    base64_str = filename.split("/")[-1].replace(".jpeg", "")
    try:
        return base64.b64decode(base64_str).decode("utf-8")
    except:
        return None


def list_files_recursive(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def write_files_to_csv(root_dir, csv_path):
    filepaths = list(list_files_recursive(root_dir))
    data = []
    for filepath in filepaths:
        decoded = decode_base64(filepath)
        data.append({"image_path": filepath, "id": decoded})
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory_to_scan> <output_csv_file>")
        sys.exit(1)
    directory = sys.argv[1]
    output_csv = sys.argv[2]
    write_files_to_csv(directory, output_csv)
