This is the README file for the code provided by CHELLY SWANN / ARTIGAS HUGO / BEN BOUAZZA ANASS.

You should put train.csv and evaluation.csv in "code/data/"

----------------------------------------------------------------------------------------------
Preprocessing : 

Make sure Tensorflow is installed on your computer.


In order to reproduce our best submission, please execute "python prepro1bis.py" and ignore all warnings
This may take some time.
This will compute the preprocessed data for both train.csv and evaluation.csv


We include two preprocessing scripts, which correspond to Preproc 1bis and Preproc 2 in our report.

- Preproc 1bis : execute "python prepro1bis.py"

- Preproc 2 : execute "python prepro2.py"
Make sure Pytorch and transformers are installed on your computer
Preproc 2 computes Preproc 1 and BERT features, and had a runtime of more than 10 hours on an 8 cores computer in order to process both train.csv and evaluation.csv
Please only run this script under Windows, since differences in the gestion of parallelization made it so we could not get it to work on UNIX operating systems 


-----------------------------------------------------------------------------------------------

Models :

In order to reproduce our best submission, please execute "python main_xgboost.py" and ignore all warnings
This will take some time, please consider running this script overnight if using your own computer
This will save the predictions in "code/data/predictions.csv"


We include two models : our XGBoost model and our neural network.

- XGBoost (Preproc 1bis needed) : execute "python main_xgboost.py"

- Neural Network (Preproc 2 needed) : execute "python main_net.py bert"
A graph representing the evolution of loss during the epochs of training will show up, you can close it without consequences
You can execute "python main_net.py no-bert" in order to ignore BERT features. This will speed up training



