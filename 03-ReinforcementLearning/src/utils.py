import csv
import numpy as np
from pathlib import Path
import os
from datetime import datetime

def load_csv(path, has_header=True):
    """
    Loads CSV file where:
        - First column is label
        - Remaining columns are features
        - First row is header

    Returns:
        X : numpy array (n_samples, n_features)
        y : numpy array (n_samples,)
    """
    X = []
    y = []

    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)

        if has_header:
            next(reader, None)  # Skip header row safely

        # for train and test, extract labels
        if ("eval" not in str(path)):

            for row in reader:
                if not row:
                    continue  # Skip empty rows

                # Convert all columns to float
                values = [float(val) for val in row]

                X.append(values[1:])
                y.append(values[0])

            return np.array(X), np.array(y)
        
        # for eval, there are no labels, only X values 
        else:

            for row in reader:
                if not row:
                    continue  # Skip empty rows

                # Convert all columns to float
                values = [float(val) for val in row]

                X.append(values)

            return np.array(X), None


def load_train_data():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/knnlsh_train.csv"
    return load_csv(DATA_DIR)


def load_test_data():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/knnlsh_test.csv"
    return load_csv(DATA_DIR)


def load_eval_data():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/knnlsh_eval.csv"
    return load_csv(DATA_DIR)

# Load Time Series data
def load_timeseries():
    """
    Loads univariate time series with date and value columns.

    Expected CSV format:
        date,value
        2020-01-01,120.3
        2020-02-01,125.6
        ...

    Returns:
        dates  : numpy array of datetime objects
        series : numpy array (n,)
    """
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    path = BASE_DIR / "data/timeseries_train.csv"
    date_format="%Y-%m-%d"
    has_header = True 

    dates = []
    values = []

    with Path(path).open("r") as f:
        reader = csv.reader(f)

        if has_header:
            next(reader, None)

        for row in reader:
            if not row:
                continue

            date_str = row[0].strip()
            value_str = row[1].strip()

            date_obj = datetime.strptime(date_str, date_format)

            dates.append(date_obj)
            values.append(float(value_str))

    return np.array(dates), np.array(values)

def load_sms_csv(filepath):
    """
    Loads CSV file.
    Assumes:
        - First column is label
        - First row is header
    Returns:
        X (numpy array), y (numpy array)
    """
    X = []
    y = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            features = row[1].strip()
            label = row[0].strip()

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

def load_train_sms():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/naivebayes_train.csv"
    return load_sms_csv(DATA_DIR)

def load_test_sms():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/naivebayes_test.csv"
    return load_sms_csv(DATA_DIR)


def load_eval_sms_csv(filepath):
    """
    Loads CSV file.
    Assumes:
        - Last column is label
        - First row is header
    Returns:
        X (numpy array), y (numpy array)
    """
    X = []
    y = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            features = row[0].strip()
            X.append(features)

    return np.array(X)

def load_eval_sms():
    BASE_DIR = Path(__file__).resolve().parents[1]
    #print (BASE_DIR)
    DATA_DIR = BASE_DIR / "data/naivebayes_eval.csv"
    return load_eval_sms_csv(DATA_DIR)

