import pandas as pd
import sklearn.datasets
import os
from typing import Dict, Optional
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Input, Activation,MaxPooling2D, AveragePooling2D, Concatenate

pd.set_option('future.no_silent_downcasting', True)

# Strings representing each dataset
HELOC_NAME = "HELOC"
CALIFORNIA_HOUSING_NAME = "California_Housing"
DENGUE_DATASET = "Dengue_Chikungunya"
COVERTYPE_NAME = "Covertype"

# Strings representing each method
IGTD_NAME = "IGTD"
REFINED_NAME = "REFINED"
BARGRAPH_NAME = "BARGRAPH"
DISTANCE_MATRIX_NAME = "DISTANCE-MATRIX"
COMBINATION_NAME = "COMBINATION"
TINTO_NAME = "TINTO"
SUPERTML_EF_NAME = "SuperTML-EF"
SUPERTML_VF_NAME = "SuperTML-VF"

DROPOUT_VALUE = 0.2

"""List with all the available datasets"""
ALL_DATASETS = [HELOC_NAME, CALIFORNIA_HOUSING_NAME, DENGUE_DATASET, COVERTYPE_NAME]

"""List with all the available image methods"""
ALL_IMAGE_METHODS = [IGTD_NAME, REFINED_NAME, BARGRAPH_NAME, DISTANCE_MATRIX_NAME, COMBINATION_NAME, TINTO_NAME, SUPERTML_EF_NAME, SUPERTML_VF_NAME]

"""Seed for reproducibility in shuffling and spliting"""
RANDOM_SEED = 1234

"""Size of test split (out of 1)"""
TEST_SPLIT_SIZE = 0.1

"""Size of validation split (out of 1)"""
VAL_SPLIT_SIZE = 0.1

"""List with the datasets that are binary classification"""
BINARY_CLASSIFICATION_DATASETS = [HELOC_NAME, DENGUE_DATASET]
assert all(c in ALL_DATASETS for c in BINARY_CLASSIFICATION_DATASETS)

"""Dictionary with pairs of dataset key and number of samples"""
N_SAMPLES_PER_DATASET: Dict[str, int] = {
    HELOC_NAME: 9_871,
    CALIFORNIA_HOUSING_NAME: 20_640,
    DENGUE_DATASET: 11_448,
    COVERTYPE_NAME: 581_012,
}
assert set(ALL_DATASETS) == set(N_SAMPLES_PER_DATASET.keys())

"""Dictionary with pairs of dataset key and 'problem' string for TINTOlib"""
DATASET_TYPES: Dict[str, str] = {
    HELOC_NAME: "supervised",
    CALIFORNIA_HOUSING_NAME: "regression",
    DENGUE_DATASET: "supervised",
    COVERTYPE_NAME: "supervised",
}
assert set(ALL_DATASETS) == set(DATASET_TYPES.keys())

DATASETS_N_CLASSES: Dict[str, Optional[int]] = {
    HELOC_NAME: 2,
    CALIFORNIA_HOUSING_NAME: None,
    DENGUE_DATASET: 2,
    COVERTYPE_NAME: 7,
}
assert set(ALL_DATASETS) == set(DATASET_TYPES.keys())

def get_X_y(dataset_name: str):
    """Given the string that represents a dataset, returns X and y"""
    if dataset_name not in ALL_DATASETS:
        raise ValueError("Cannot find a dataset with that name.")

    elif dataset_name == HELOC_NAME:
        path_to_dataset = os.path.join("datasets", "heloc.csv")
        df = pd.read_csv(path_to_dataset)
        label_col = "RiskPerformance"
        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].replace({"Good": 0, "Bad": 1}).astype(int).to_numpy()

    elif dataset_name == COVERTYPE_NAME:
        X,y = sklearn.datasets.fetch_covtype(return_X_y=True)
        y = y-1

    elif dataset_name == CALIFORNIA_HOUSING_NAME:
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif dataset_name == DENGUE_DATASET:
        path_to_dataset = os.path.join("datasets", "arboviruses_dataset.csv")
        df = pd.read_csv(path_to_dataset, sep=";")

        label_col = "CLASSI_FIN"
        desired_targets = ['CHIKUNGUNYA', 'DENGUE']
        df = df[df[label_col].isin(desired_targets)]

        X = df.drop(label_col, axis=1).to_numpy()
        y = sklearn.preprocessing.LabelEncoder().fit_transform(df[label_col])

    else:
        raise NotImplementedError()

    assert X.shape[0] == N_SAMPLES_PER_DATASET[dataset_name]
    return X,y

def get_indices_train_eval(dataset_name: str):
    total_elems: int = N_SAMPLES_PER_DATASET.get(dataset_name, -1)
    if total_elems == -1:
        raise ValueError("Cannot find a dataset with that name.")

    # Create the array with all the indices
    indices = np.arange(total_elems)

    # Remove the indices from the eval
    indices_test = get_indices_test(dataset_name)
    non_test_indices = np.setdiff1d(indices, indices_test)
    assert indices_test.shape[0] + non_test_indices.shape[0] == total_elems

    # Split the remaining
    r = np.random.default_rng(seed=RANDOM_SEED)
    r.shuffle(non_test_indices)

    n_val = int(total_elems * VAL_SPLIT_SIZE)
    train_indices, val_indices = non_test_indices[:-n_val], non_test_indices[-n_val:]
    assert train_indices.shape[0] + val_indices.shape[0] == non_test_indices.shape[0]
    return train_indices, val_indices

def get_indices_train1_eval1_train2_eval2(dataset_name: str):
    total_elems: int = N_SAMPLES_PER_DATASET.get(dataset_name, -1)
    if total_elems == -1:
        raise ValueError("Cannot find a dataset with that name.")

    # Get train_val indices
    train_indices, val_indices = get_indices_train_eval(dataset_name)
    # Split train_val indices in half
    train_1, train_2 = train_indices[:train_indices.shape[0]//2], train_indices[train_indices.shape[0]//2:]
    val_1, val_2 = val_indices[:val_indices.shape[0]//2], val_indices[val_indices.shape[0]//2:]

    # Return
    assert train_1.shape[0] + val_1.shape[0] + train_2.shape[0] + val_2.shape[0] == train_indices.shape[0] + val_indices.shape[0]
    return train_1, val_1, train_2, val_2

def get_indices_test(dataset_name: str):
    total_elems: int = N_SAMPLES_PER_DATASET.get(dataset_name, -1)
    if total_elems == -1:
        raise ValueError("Cannot find a dataset with that name.")

    # Create the array with all the indices
    indices = np.arange(total_elems)

    # Shuffle the indices
    r = np.random.default_rng(seed=RANDOM_SEED)
    r.shuffle(indices)

    # Return the N first
    n = int(total_elems * TEST_SPLIT_SIZE)
    return indices[:n]

def get_results_path():
    p = "results"
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_images_base_path():
    p = "images"
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_images_path_for_dataset(dataset_name, image_method):
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    if image_method not in ALL_IMAGE_METHODS:
        raise Exception("This image method does not exist")
    p = os.path.join(get_images_base_path(), dataset_name, image_method)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_classicdescriptors_path(dataset_name):
    def get_classicdescriptors_base_path():
        p = os.path.join(get_results_path(), "classic_descriptors")
        if not os.path.exists(p):
            os.makedirs(p)
        return p
    
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    p = os.path.join(get_classicdescriptors_base_path(), dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    return p


def get_classicdescriptors_split1_path(dataset_name):
    def get_classicdescriptors_split1_base_path():
        p = os.path.join(get_results_path(), "classic_descriptors_split1")
        if not os.path.exists(p):
            os.makedirs(p)
        return p
    
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    p = os.path.join(get_classicdescriptors_split1_base_path(), dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_cnnmodels_path(dataset_name):
    def get_cnnmodels_base_path():
        p = os.path.join(get_results_path(), "cnn_models")
        if not os.path.exists(p):
            os.makedirs(p)
        return p
    
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    p = os.path.join(get_cnnmodels_base_path(), dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_cnnmlp_models_path(dataset_name):
    def get_cnnmlp_models_base_path():
        p = os.path.join(get_results_path(), "cnn_mlp_models")
        if not os.path.exists(p):
            os.makedirs(p)
        return p
    
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    p = os.path.join(get_cnnmlp_models_base_path(), dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_cnn_classicdescriptors_path(dataset_name):
    def get_cnn_classicdescriptors_base_path():
        p = os.path.join(get_results_path(), "cnn_classic_models")
        if not os.path.exists(p):
            os.makedirs(p)
        return p
    
    if dataset_name not in ALL_DATASETS:
        raise Exception("This dataset does not exist")
    p = os.path.join(get_cnn_classicdescriptors_base_path(), dataset_name)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

def get_dataset_type_str(dataset_name: str):
    dataset_type_str = DATASET_TYPES.get(dataset_name, None)
    if dataset_type_str is None:
        raise ValueError("Cannot find a dataset with that name.")
    return dataset_type_str

def is_dataset_classification(dataset_name: str):
    return get_dataset_type_str(dataset_name) == "supervised"

def is_dataset_multiclass_classification(dataset_name: str):
    return is_dataset_classification(dataset_name) and dataset_name not in BINARY_CLASSIFICATION_DATASETS

def is_dataset_binary_classification(dataset_name: str):
    return is_dataset_classification(dataset_name) and dataset_name in BINARY_CLASSIFICATION_DATASETS

def is_dataset_regression(dataset_name: str):
    return get_dataset_type_str(dataset_name) == "regression"

def get_number_of_classes(dataset_name: str):
    n = DATASETS_N_CLASSES.get(dataset_name)
    if not is_dataset_classification(dataset_name):
        assert n is None
        raise Exception("This dataset is not classification")
    elif is_dataset_binary_classification(dataset_name):
        assert n==2
        return n
    elif is_dataset_multiclass_classification(dataset_name):
        assert n>2
        return n
    raise Exception("This shouldn't have happened.")

def get_cnn_branch(input_shape):
    """Returns the CNN branches"""
    # Input layer
    input_branch = Input(input_shape)

    # Start branch 1
    tower_1 = Conv2D(16, (3,3), activation='relu',padding="same")(input_branch)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = MaxPooling2D(2,2)(tower_1)
    tower_1 = Dropout(DROPOUT_VALUE)(tower_1)

    tower_1 = Conv2D(32, (3,3), activation='relu',padding="same")(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = MaxPooling2D(2,2)(tower_1)
    tower_1 = Dropout(DROPOUT_VALUE)(tower_1)

    tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = MaxPooling2D(2,2)(tower_1)
    tower_1 = Dropout(DROPOUT_VALUE)(tower_1)

    tower_1 = Conv2D(64, (3,3), activation='relu',padding="same")(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation('relu')(tower_1)
    tower_1 = MaxPooling2D(2,2)(tower_1)
    tower_1 = Dropout(DROPOUT_VALUE)(tower_1)
    # End branch 1

    # Start branch 2
    tower_2 = Conv2D(16, (5,5), activation='relu',padding="same")(input_branch)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = AveragePooling2D(2,2)(tower_2)
    tower_2 = Dropout(DROPOUT_VALUE)(tower_2)

    tower_2 = Conv2D(32, (5,5), activation='relu',padding="same")(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = AveragePooling2D(2,2)(tower_2)
    tower_2 = Dropout(DROPOUT_VALUE)(tower_2)

    tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = AveragePooling2D(2,2)(tower_2)
    tower_2 = Dropout(DROPOUT_VALUE)(tower_2)

    tower_2 = Conv2D(64, (5,5), activation='relu',padding="same")(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation('relu')(tower_2)
    tower_2 = AveragePooling2D(2,2)(tower_2)
    tower_2 = Dropout(DROPOUT_VALUE)(tower_2)
    # End branch 2

    # Concatenate the 2 branches
    merged = Concatenate(axis=1)([tower_1, tower_2])

    # Dense layers
    out = Flatten()(merged)
    out = Dense(256, activation='relu')(out)
    out = Dense(128, activation='sigmoid')(out)
    out = Dense(64, activation='sigmoid')(out)
    out = Dense(32, activation='sigmoid')(out)

    return (input_branch, out)

def get_mlp_branch(input_shape):
    """Returns the MLP branch"""
    input_ffnn = Input(shape=input_shape)

    x = Dense(64, activation="relu")(input_ffnn)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_VALUE)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_VALUE)(x)

    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_VALUE)(x)

    return (input_ffnn, x)

# Check
for dataset_name in ALL_DATASETS:
    assert is_dataset_classification(dataset_name) ^ is_dataset_regression(dataset_name)