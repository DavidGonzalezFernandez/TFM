import utils
import pandas as pd
import os
import numpy as np
import shutil

from TINTOlib.barGraph import BarGraph
from TINTOlib.combination import Combination
from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.tinto import TINTO

pd.set_option('future.no_silent_downcasting', True)

datasets = {
    "heloc": "HELOC",
    "arbo_viruses": "Arboviruses",
    "california_housing": "CaliforniaHousing",
    "adult_income": "AdultIncome",
    "covertype": "Covertype",
    "higgs": "HIGGS",
}

dataset_type = {
    "heloc": "supervised",
    "adult_income": "supervised",
    "covertype": "supervised",
    "california_housing": "regression",
    "arbo_viruses": "supervised",
    "higgs": "supervised",
}

dataset_scale = {
    "heloc": [5,5],
    "adult_income": [4,4],
    "covertype": [8,8],
    "california_housing": [3,3],
    "arbo_viruses": [6,6],
    "higgs": [6,6],
}


def main():
    base_path = "images"

    if base_path not in os.listdir():
        os.mkdir(base_path)

    create_igtd(base_path)
    create_refined(base_path)
    create_barGraph(base_path)
    create_combination(base_path)
    create_distanceMatrix(base_path)
    create_tinto(base_path)

def create_tinto(base_path):
    image_folder = "tinto"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating tinto")

    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = TINTO(
            problem=dataset_type[dataset_key],
            verbose=False,
            pixels = dataset_scale[dataset_key][0]
        )

        image_model.generateImages(df,final_folder)

def create_distanceMatrix(base_path):
    image_folder = "distancematrix"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating distancematrix")

    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = DistanceMatrix(
            verbose=False,
            scale=dataset_scale[dataset_key],
            problem=dataset_type[dataset_key]
        )

        image_model.generateImages(df,final_folder)

def create_combination(base_path):
    image_folder = "combination"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating combination")

    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = Combination(
            verbose=False,
            pixel_width=1,
            gap=1,
            problem=dataset_type[dataset_key]
        )

        image_model.generateImages(df,final_folder)

def create_barGraph(base_path):
    image_folder = "bargraph"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating bargraph")

    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = BarGraph(
            verbose=False,
            pixel_width=1,
            gap=1,
            problem=dataset_type[dataset_key]
        )

        image_model.generateImages(df,final_folder)

def create_refined(base_path):
    image_folder = "refined"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating REFINED")

    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = REFINED(
            problem=dataset_type[dataset_key],
            save_image_size = None,
            verbose=False
        )

        image_model.generateImages(df,final_folder)



def create_igtd(base_path):
    image_folder = "igtd"
    if image_folder not in os.listdir(base_path):
        base_path = os.path.join(base_path, image_folder)
        os.mkdir(base_path)
    
    print("Creating IGTD")

    # Iterate over dataset
    for dataset_key in datasets.keys():
        print(f"\t{dataset_key}")

        dataset_folder = datasets[dataset_key].lower()
        final_folder = os.path.join(base_path, dataset_folder)
        if dataset_folder in os.listdir(base_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(datasets[dataset_key])
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = IGTD(
            problem=dataset_type[dataset_key],
            save_image_size = None,
            scale = dataset_scale[dataset_key],
            verbose=False
        )

        image_model.generateImages(df,final_folder)


if __name__ == "__main__":
    main()