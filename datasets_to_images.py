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

from utils import HELOC_NAME, ADULT_INCOME_NAME, HIGGS_NAME, COVERTYPE_NAME, CALIFORNIA_HOUSING_NAME, ARBOVIRUSES_NAME


pd.set_option('future.no_silent_downcasting', True)

datasets = [HELOC_NAME, ADULT_INCOME_NAME, HIGGS_NAME, COVERTYPE_NAME, CALIFORNIA_HOUSING_NAME, ARBOVIRUSES_NAME]

dataset_scale = {
    HELOC_NAME: [5,5],
    ADULT_INCOME_NAME: [4,4],
    COVERTYPE_NAME: [8,8],
    CALIFORNIA_HOUSING_NAME: [3,3],
    ARBOVIRUSES_NAME: [6,6],
    HIGGS_NAME: [6,6],
}


def main():
    base_path = utils.get_images_path()

    if base_path not in os.listdir():
        os.mkdir(base_path)

    create_igtd(base_path)
    create_refined(base_path)
    # create_barGraph(base_path)          # TODO: check
    # create_combination(base_path)       # TODO: check   
    # create_distanceMatrix(base_path)    # TODO: check
    # TODO: add SUPERTML
    create_tinto(base_path)

def create_tinto(base_path):
    image_folder = "tinto"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating tinto")

    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = TINTO(
            problem=utils.get_dataset_type(dataset_name),
            verbose=False,
            pixels = dataset_scale[dataset_name][0]
        )

        image_model.generateImages(df,final_folder)

def create_distanceMatrix(base_path):
    image_folder = "distancematrix"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating distancematrix")

    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = DistanceMatrix(
            verbose=False,
            scale=dataset_scale[dataset_name],
            problem=utils.get_dataset_type(dataset_name)
        )

        image_model.generateImages(df,final_folder)

def create_combination(base_path):
    image_folder = "combination"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating combination")

    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = Combination(
            verbose=False,
            pixel_width=1,
            gap=1,
            problem=utils.get_dataset_type(dataset_name)
        )

        image_model.generateImages(df,final_folder)

def create_barGraph(base_path):
    image_folder = "bargraph"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating bargraph")

    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = BarGraph(
            verbose=False,
            pixel_width=1,
            gap=1,
            problem=utils.get_dataset_type(dataset_name)
        )

        image_model.generateImages(df,final_folder)

def create_refined(base_path):
    image_folder = "refined"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating REFINED")

    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = REFINED(
            problem=utils.get_dataset_type(dataset_name),
            verbose=False
        )

        image_model.generateImages(df,final_folder)



def create_igtd(base_path):
    image_folder = "igtd"
    image_path = os.path.join(base_path, image_folder)
    if image_folder not in os.listdir(base_path):
        os.mkdir(image_path)
    
    print("Creating IGTD")

    # Iterate over dataset
    for dataset_name in datasets:
        print(f"\t{dataset_name}")

        dataset_folder = dataset_name.lower()
        final_folder = os.path.join(image_path, dataset_folder)
        if dataset_folder in os.listdir(image_path):
            if any(f.endswith(".csv") for f in os.listdir(final_folder)):
                continue
            else:
                shutil.rmtree(dataset_folder)

        X,y = utils.get_X_y(dataset_name)
        df = pd.DataFrame(np.column_stack ((X,y)))

        image_model = IGTD(
            problem=utils.get_dataset_type(dataset_name),
            scale = dataset_scale[dataset_name],
            verbose=False
        )

        image_model.generateImages(df,final_folder)


if __name__ == "__main__":
    main()