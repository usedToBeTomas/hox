from tqdm import tqdm
import os
import urllib.request
import gzip
import shutil
import numpy as np

#Load mnist dataset
def mnist():
    print("Loading MNIST dataset...")
    print("Checking...")
    dataset_dir="./MNIST"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    missing_files = []
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    if missing_files:
        print("Downloading missing files...")
        for file in missing_files:
            print(f"Downloading {file}...")
            url = base_url + file
            urllib.request.urlretrieve(url, os.path.join(dataset_dir, file))
            print("Download completed.")
    else:
        print("All files are already downloaded.")
    print("Preparing dataset...")
    X = (np.frombuffer(gzip.open("MNIST/train-images-idx3-ubyte.gz", "rb").read(), dtype=np.uint8)[16:].reshape((-1, 784)) / 255).astype(np.float32)
    Y = (np.eye(10)[np.frombuffer(gzip.open("MNIST/train-labels-idx1-ubyte.gz", "rb").read(), dtype=np.uint8)[8:]]).astype(np.float32)
    x = np.frombuffer(gzip.open("MNIST/t10k-images-idx3-ubyte.gz", "rb").read(), dtype=np.uint8)[16:].reshape((-1, 784)) / 255
    y = np.frombuffer(gzip.open("MNIST/t10k-labels-idx1-ubyte.gz", "rb").read(), dtype=np.uint8)[8:]
    print("MNIST dataset loaded.")
    return X, Y, x, y
