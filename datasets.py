import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import os, zipfile

uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
datasets = {"boston": 
                {"fpath": "data/uci/boston.data",
                 "url": uci_url + "housing/housing.data"},
            "concrete":
                {"fpath": "data/uci/concrete.xls",
                 "url": uci_url + "concrete/compressive/Concrete_Data.xls"},
            "energy":
                {"fpath": "data/uci/energy.xlsx",
                 "url": uci_url + "00242/ENB2012_data.xlsx"},
            "kin8nm":
                {"fpath": "data/uci/kin8nm.csv",
                 "url": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"},
            "naval":
                {"fpath": "data/uci/naval.zip",
                 "url": uci_url + "00316/UCI%20CBM%20Dataset.zip"},
            "power":
                {"fpath": "data/uci/power.zip",
                 "url": uci_url + "00294/CCPP.zip"},
            "protein":
                {"fpath": "data/uci/protein.csv",
                 "url": uci_url + "00265/CASP.csv"},
            "wine":
                {"fpath": "data/uci/wine.csv",
                 "url": uci_url + "wine-quality/winequality-red.csv"},
            "yacht":
                {"fpath": "data/uci/yacht.data",
                 "url": uci_url + "/00243/yacht_hydrodynamics.data"}
            }

def download(url, fpath):
    """Download and unzip data as necessary"""
    response = requests.get(url, stream=True, verify=False)
    print(f"Downloading '{url}'...")
    if response.status_code != 200:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(fpath, 'wb') as output_file:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            output_file.write(data)
    progress_bar.close()
    print(f"Download complete. Saved as '{fpath}'.")
    if fpath.endswith(".zip"):
        data_dir = os.path.dirname(fpath)
        with zipfile.ZipFile(fpath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Data unzipped in '{data_dir}'.")

def autoload_csv(fpath):
    """Load a csv file, automatically handling headers and delimiters"""
    possible_delimiters = [',', ';', ' ', '\t']
    for delimiter in possible_delimiters:
        try:
            df = pd.read_csv(fpath, sep=delimiter, engine="python")
            return df.to_numpy()
        except pd.errors.ParserError:
            continue
    raise ValueError("Unable to detect csv delimiter.")

def generate_uci_data(name):
    """Return numpy array of desired dataset"""
    name = name.lower().strip()
    fpath = datasets[name]["fpath"]
    url = datasets[name]["url"]
    if not os.path.isfile(fpath):
        data_dir = os.path.dirname(fpath)
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(data_dir)
        download(url, fpath)
    
    if name == "boston":
        data = pd.read_fwf(fpath, header=None).to_numpy()
    elif name == "concrete":
        data = pd.read_excel(fpath).to_numpy()
    elif name == "energy":
        data = pd.read_excel(fpath).to_numpy()
        data = data[:, :-1]
    elif name == "kin8nm":
        data = autoload_csv(fpath)
    elif name == "naval":
        fpath = "data/uci/UCI CBM Dataset/data.txt"
        data = pd.read_fwf(fpath, header=None).to_numpy()
        data = data[:, :-1] # Truncate output
        data = np.delete(data, [8, 11], axis=1) # 0-std features
    elif name == "power":
        fpath = "data/uci/CCPP/Folds5x2_pp.xlsx"
        data = pd.read_excel(fpath).to_numpy()
    elif name == "protein":
        data = autoload_csv(fpath)
        data[:, [0, -1]] = data[:, [-1, 0]] # Swap output column
    elif name == "wine":
        data = autoload_csv(fpath)
    elif name == "yacht":
        data = pd.read_fwf(fpath, header=None).to_numpy()
    else:
        print(f"Unsupported dataset: '{name}'")
        n_samples = input_size = data = None
        return n_samples, input_size, data
    data = np.float32(data)
    n_samples, input_size = data.shape
    input_size -= 1
    return n_samples, input_size, data

    # if name == "naval":
    #     fpath = "data/uci/UCI CBM Dataset/data.txt"
    # elif name == "power":
    #     fpath = "data/uci/CCPP/Folds5x2_pp.xlsx"

    # ext = os.path.splitext(fpath)[1]
    # if name == "boston":
    #     data = pd.read_fwf(fpath, header=None).to_numpy()
    #     data = np.float32(data)
    #     n_samples, input_size = data.shape
    #     input_size -= 1
    # elif name == "naval":
    #     data = pd.read_fwf(fpath, header=None).to_numpy()
    #     data = data[:, :-1] # Truncate output
    #     data = np.delete(data, [8, 11], axis=1) # 0-std features
    #     data = np.float32(data)
    #     n_samples, input_size = data.shape
    #     input_size -= 1
    # elif name == "yacht":
    #     data = pd.read_fwf(fpath, header=None).to_numpy()
    #     data = np.float32(data)
    #     n_samples, input_size = data.shape
    #     input_size -= 1
    # elif ext == ".xls" or ext == ".xlsx":
    #     data = pd.read_excel(fpath).to_numpy()
    #     data = np.float32(data)
    #     n_samples, input_size = data.shape
    #     input_size -= 1
    # elif ext == ".csv" or ext == ".txt":
    #     #data = np.loadtxt(fpath, dtype=np.float32)
    #     data = autoload_csv(fpath)
    #     if name == "energy":
    #         data = data[:, :-1] # Truncate output
    #     elif name == "protein":
    #         data[:, [0, -1]] = data[:, [-1, 0]] # Swap output column
    #     data = np.float32(data)
    #     n_samples, input_size = data.shape
    #     input_size -= 1
    # else:
    #     print(f"Unsupported file type: {ext}")
    #     n_samples = input_size = data = None
    # return n_samples, input_size, data


# def generate_data(url, fpath):
#     """Download data from url if not in fpath"""
#     if not os.path.isfile(fpath):
#         data_dir = os.path.dirname(fpath)
#         if not os.path.exists(os.path.dirname(fpath)):
#             os.makedirs(data_dir)
#         download(url, fpath)