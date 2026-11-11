import os
import urllib.request
import zipfile


os.makedirs("data", exist_ok=True)
url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

zip_path = "data/ml-latest-small.zip"


if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data")
