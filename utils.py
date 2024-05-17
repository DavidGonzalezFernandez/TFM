import pandas as pd
import sklearn
import sklearn.datasets
import os
from typing import Dict
import numpy as np

HELOC_NAME = "HELOC"
ADULT_INCOME_NAME = "AdultIncome"
HIGGS_NAME = "HIGGS"
COVERTYPE_NAME = "Covertype"
CALIFORNIA_HOUSING_NAME = "CaliforniaHousing"
ARBOVIRUSES_NAME = "Arboviruses"

RANDOM_SEED = 1234

TEST_SPLIT_SIZE = 0.1
VAL_SPLIT_SIZE = 0.1

n_samples_per_dataset: Dict[str, int] = {
    HELOC_NAME: 10459,
    ADULT_INCOME_NAME: 32561,
    HIGGS_NAME: 11000000,
    COVERTYPE_NAME: 581012,
    CALIFORNIA_HOUSING_NAME: 20640,
    ARBOVIRUSES_NAME: 17172,
}

def get_X_y(dataset_name: str):
    if dataset_name == HELOC_NAME:
        # I downloaded the CSV (pero se puede sacar lo mismo de dataset = load_dataset("mstz/heloc"))
        path_to_dataset = os.path.join("datasets", "heloc_dataset_v1.csv")
        df = pd.read_csv(path_to_dataset)
        df = df.dropna()
        label_col = "RiskPerformance"
        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].replace({"Good":0, "Bad": 1}).astype(int).to_numpy()
    
    elif dataset_name == ADULT_INCOME_NAME:
        # Lo descargo directamente de la página
        path_to_dataset = os.path.join("datasets", "adult.data.txt")
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(path_to_dataset, names=columns)
        df = df.dropna()

        string_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

        for columna in string_columns:
            le = sklearn.preprocessing.LabelEncoder()
            df[columna] = le.fit_transform(df[columna])

        X = df[features].to_numpy()
        y = df[label].replace({' >50K': 0, ' <=50K': 1}).astype(int).to_numpy()
    
    elif dataset_name == HIGGS_NAME:
        # Se puede descargar el archivo https://archive.ics.uci.edu/dataset/280/higgs
        path_to_dataset = os.path.join("datasets", "HIGGS.csv.gz")
        df = pd.read_csv(path_to_dataset, header=None)
        df = df.dropna()

        """The first column is the class label (1 for signal, 0 for background),
        followed by the 28 features (21 low-level features then 7 high-level features):
        lepton  pT, lepton  eta, lepton  phi, missing energy magnitude, missing energy phi,
        jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi,
        jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta,
        jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb."""

        df.columns = [
            "class_label", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
            "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag", "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
            "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag", "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
            "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
        ]
        label_col = df.columns[0]

        X = df.drop(label_col, axis=1).replace({0.0: 0, 1.0: 1}).to_numpy()
        y = df[label_col].to_numpy()
    
    elif dataset_name == COVERTYPE_NAME:
        X,y = sklearn.datasets.fetch_covtype(return_X_y=True)
    
    elif dataset_name == CALIFORNIA_HOUSING_NAME:
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    
    elif dataset_name == ARBOVIRUSES_NAME:
        # Se puede descargar el dataset desde
        path_to_dataset = os.path.join("datasets", "arboviruses_dataset.csv")
        df = pd.read_csv(path_to_dataset, sep=";")
        df = df.dropna()
        label_col = "CLASSI_FIN"
        X = df.drop(label_col, axis=1).to_numpy()

        le = sklearn.preprocessing.LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])

        y = df[label_col].to_numpy()
    
    else:
        raise ValueError("Cannot find a dataset with that name.")
    
    assert X.shape[0] == n_samples_per_dataset[dataset_name]
    return X,y

def get_indices_train_eval(dataset_name: str):
    total_elems: int = n_samples_per_dataset.get(dataset_name, -1)
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
    total_elems: int = n_samples_per_dataset.get(dataset_name, -1)
    if total_elems == -1:
        raise ValueError("Cannot find a dataset with that name.")

    # Create the array with all the indices
    indices = np.arange(total_elems)

    # Remove the indices from eval
    indices_test = get_indices_test(dataset_name)
    non_test_indices = np.setdiff1d(indices, indices_test)
    assert indices_test.shape[0] + non_test_indices.shape[0] == total_elems

    # Split 50% for 1 and 50% for 2
    split_size = non_test_indices.shape[0] // 2
    split_1, split_2 = non_test_indices[:split_size], non_test_indices[split_size:]
    assert split_1.shape[0] + split_2.shape[0] == non_test_indices.shape[0]

    r = np.random.default_rng(seed=RANDOM_SEED)
    n_val = int(total_elems * VAL_SPLIT_SIZE)//2

    # Get train 1 and eval 1
    r.shuffle(split_1)
    train_1, val_1 = split_1[:-n_val], split_1[n_val:]
    assert train_1.shape[0] + val_1.shape[0] == split_1.shape[0]

    # Get train2 and eval 2
    r.shuffle(split_2)
    train_2, val_2 = split_2[:-n_val], split_2[n_val:]
    assert train_2.shape[0] + val_2.shape[0] == split_2.shape[0]

    # Return
    assert train_1.shape[0] + val_1.shape[0] + train_2.shape[0] + val_2.shape[0] == non_test_indices.shape[0]
    return train_1, val_1, train_2, val_2

def get_indices_test(dataset_name: str):
    total_elems: int = n_samples_per_dataset.get(dataset_name, -1)
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
