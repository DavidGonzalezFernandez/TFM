import utils
import pandas as pd
import os
import numpy as np
import shutil

from TINTOlib.igtd import IGTD
from TINTOlib.refined import REFINED
from TINTOlib.barGraph import BarGraph
from TINTOlib.distanceMatrix import DistanceMatrix
from TINTOlib.combination import Combination
from TINTOlib.tinto import TINTO
from TINTOlib.supertml import SuperTML

from tqdm import tqdm

from utils import HELOC_NAME, CALIFORNIA_HOUSING_NAME, COVERTYPE_NAME, DENGUE_DATASET
from utils import IGTD_NAME, REFINED_NAME, BARGRAPH_NAME, DISTANCE_MATRIX_NAME, COMBINATION_NAME, TINTO_NAME, SUPERTML_EF_NAME, SUPERTML_VF_NAME
from utils import ALL_DATASETS, ALL_IMAGE_METHODS

DATASET_ZOOM = {
    HELOC_NAME: 5,
    CALIFORNIA_HOUSING_NAME: 10,
    DENGUE_DATASET: 4,
    COVERTYPE_NAME: 3,
}
assert set(DATASET_ZOOM.keys()) == set(ALL_DATASETS)

DATASET_SQRT_COLUMNS = {
    HELOC_NAME: 5,
    CALIFORNIA_HOUSING_NAME: 3,
    DENGUE_DATASET: 6,
    COVERTYPE_NAME: 8,
}
assert set(DATASET_SQRT_COLUMNS.keys()) == set(ALL_DATASETS)

DATASET_SUPERTML_EF_TEXT_SIZES = {
    HELOC_NAME: 5,
    CALIFORNIA_HOUSING_NAME: 7,
    DENGUE_DATASET: 5,
    COVERTYPE_NAME: 4,
}
assert set(DATASET_SUPERTML_EF_TEXT_SIZES.keys()) == set(ALL_DATASETS)

DATASET_SUPERTML_VF_TEXT_SIZES = {
    HELOC_NAME: 25,
    CALIFORNIA_HOUSING_NAME: 35,
    DENGUE_DATASET: 40,
    COVERTYPE_NAME: 40,
}
assert set(DATASET_SUPERTML_EF_TEXT_SIZES.keys()) == set(ALL_DATASETS)

def dataset_to_igtd(dataset_name):
    image_method_name = IGTD_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = IGTD(
        problem=utils.get_dataset_type_str(dataset_name),
        scale = [DATASET_SQRT_COLUMNS[dataset_name], DATASET_SQRT_COLUMNS[dataset_name]],
        zoom=DATASET_ZOOM[dataset_name]
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_refined(dataset_name):
    image_method_name = REFINED_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = REFINED(
        problem=utils.get_dataset_type_str(dataset_name),
        zoom=DATASET_ZOOM[dataset_name]
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_bargraph(dataset_name):
    image_method_name = BARGRAPH_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = BarGraph(
        problem=utils.get_dataset_type_str(dataset_name),
        gap=2
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_distancematrix(dataset_name):
    image_method_name = DISTANCE_MATRIX_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = DistanceMatrix(
        problem=utils.get_dataset_type_str(dataset_name),
        zoom=DATASET_ZOOM[dataset_name]
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_combination(dataset_name):
    image_method_name = COMBINATION_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = Combination(
        problem=utils.get_dataset_type_str(dataset_name),
        zoom=DATASET_ZOOM[dataset_name]
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_tinto(dataset_name):
    image_method_name = TINTO_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = TINTO(
        problem=utils.get_dataset_type_str(dataset_name),
        pixels=20
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_supertml_ef(dataset_name):
    image_method_name = SUPERTML_EF_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = SuperTML(
        problem=utils.get_dataset_type_str(dataset_name),
        feature_importance=False,
        font_size = DATASET_SUPERTML_EF_TEXT_SIZES[dataset_name],
        pixels = 112
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_supertml_vf(dataset_name):
    image_method_name = SUPERTML_VF_NAME
    p = utils.get_images_path_for_dataset(dataset_name, image_method_name)
    if any(f.endswith(".csv") for f in os.listdir(p)):
        # Already created
        return
    
    # Remove folder with images from previous runs
    shutil.rmtree(p)

    image_model = SuperTML(
        problem=utils.get_dataset_type_str(dataset_name),
        feature_importance=True,
        font_size = DATASET_SUPERTML_VF_TEXT_SIZES[dataset_name],
        pixels = 112
    )
    X,y = utils.get_X_y(dataset_name)
    df = pd.DataFrame(np.column_stack ((X,y)))
    image_model.generateImages(
        df,
        utils.get_images_path_for_dataset(dataset_name, image_method_name)
    )

def dataset_to_images(dataset_name):
    functions_to_use = [
        dataset_to_igtd,
        dataset_to_refined,
        dataset_to_bargraph,
        dataset_to_distancematrix,
        dataset_to_combination,
        dataset_to_tinto,
        dataset_to_supertml_ef,
        dataset_to_supertml_vf
    ]
    assert len(ALL_IMAGE_METHODS) == len(functions_to_use)
    
    print(dataset_name)
    for f in tqdm(functions_to_use, total=len(functions_to_use)):
        f(dataset_name)


def main():
    for dataset_name in ALL_DATASETS:
        dataset_to_images(dataset_name)


if __name__ == "__main__":
    main()
