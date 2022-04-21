import pandas as pd
import logging
from src.preprocessor import *
from multiprocessing import Pool

# We first read the data then we transform the columns for both train.csv and evaluation.csv

if __name__ == '__main__':

    data = pd.read_csv('data/train.csv')

    preproc_data = preprocessor(table = data)
    preproc_data.run('data/preprocessed_train.csv')

    data = pd.read_csv('data/evaluation.csv')

    preproc_data = preprocessor(table = data)
    preproc_data.run('data/preprocessed_evaluation.csv')

