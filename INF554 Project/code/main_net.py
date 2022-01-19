import sys
from src.net import *
from src.constants import *

# We first read the data and transform the columns, then run our neural network 

if __name__ == '__main__':

    data = pd.read_csv('data/preprocessed_train.csv')
    bert = True

    if sys.argv[1] == 'no-bert' :
        bert = False

    neural = Net(data,20)
    neural.run_model(bert = bert,saving_path='model/model_2',eval = True)
    
