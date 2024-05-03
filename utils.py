import pandas as pd
import sklearn
import sklearn.datasets
import os

HELOC_NAME = "HELOC"
ADULT_INCOME_NAME = "AdultIncome"
HIGGS_NAME = "HIGGS"
COVERTYPE_NAME = "Covertype"
CALIFORNIA_HOUSING_NAME = "CaliforniaHousing"
ARBOVIRUSES_NAME = "Arboviruses"

def load_data(
    dataset_name      
):
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
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)
        df = df.dropna()
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

        X = df.drop(label_col, axis=1).to_numpy()
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
        y = df[label_col].to_numpy()
    
    else:
        raise ValueError("Cannot find a dataset with that key.")
    
    return X,y