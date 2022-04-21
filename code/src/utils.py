import numpy as np 
import pandas as pd

def remove_char(string,char='.'):
    """
    Take the first part of a string from its begining to the first appearing of char

    Input : 
        string : str, 
            string that must be processed
        char : str, default = '.',
            character at which we want to cut the string
    Output : 
        A part of the input string. 
    """
    char_pos = string.find(char)
    if char_pos > 0 :
        return string[:char_pos]
    else : 
        return string


def batch_merger(df_1,df_2,saving_path,left_on, right_on,columns,batch_size = 100):
    """
    This is a batch merger that enable us to merge to files that are very big by writting merging them two by two and 
    writing them in a csv file.

    Inputs : 
        df_1 : pandas.DataFrame, 
            First df to be merged.
        df_2 : pandas.DataFrame, 
            Second df to be merged.
        saving_path : str, 
            path where we save the merge DataFrame
        left_on : str, 
            key column of df_1 
        right_on : str, 
            key column of df_2
        batch_size : int , default = 100, 
            size of the batch 
    Ouput : 
        None : The merge file will be saved into the saving path batch by batch
    """

    header = pd.DataFrame(columns = columns)
    header.to_csv(saving_path,index = False, header = True)

    batch = np.arange(len(df_1),step = 100) 
    for i in range(len(batch)-1):
        partial_df = df_1[batch[i]:batch[i+1]].merge(df_2[batch[i]:batch[i+1]], left_on=left_on,right_on=right_on)
        partial_df.to_csv(saving_path,mode='a',header=False, index = False)
    # Save last batch which may have different size from the other batches
    partial_df = df_1[batch[-1]:].merge(df_2[batch[-1]:], left_on= left_on, right_on=right_on)
    partial_df.to_csv(saving_path,mode = 'a',header=False, index = False)

def hstg_len(x):
    array = x.split()
    if array[0]=='nan':
        return 0
    else :
        return len(array)


    



