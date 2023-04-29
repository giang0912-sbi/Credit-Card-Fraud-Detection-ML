import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the credit card fraud dataset from the file path
    :param file_path: string, the path of the input file
    :return: pandas dataframe, the loaded data
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the credit card fraud dataset
    :param data: pandas dataframe, the loaded data
    :return: pandas dataframe, the preprocessed data
    """
    # drop unnecessary columns
    data = data.drop(['Time'], axis=1)

    # scale the amount column
    data['Amount'] = np.log(data['Amount'] + 1)

    # normalize the data
    for col in data.columns:
        if col not in ['Class']:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data
