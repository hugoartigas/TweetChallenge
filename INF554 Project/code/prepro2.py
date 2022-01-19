#####################################
############ Preprocessing ##########
#####################################

# This python file is dedicated to Preprocessing of BERT features. It will take several hours

from scipy.sparse import construct
from src.utils import *
from src.constants import *
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch as pt
import concurrent.futures as cf
import multiprocessing as mlt


def columns_preprocessing(df):

    """
    Preprocessing the columns. 
    Interpreting timestamps as date, day of week and hours away from noon
    Bool columns are going to be change into 1 and 0. 
    String columns are setting into lower columns, nan are being changed into string.
    We strip all meaningless characters from the text
    """

    df['date'] = pd.to_datetime(df['timestamp'],unit='ms')
    df['day'] = df['date'].dt.dayofweek.astype('category')
    df['time'] = np.abs(df['date'].dt.hour + df['date'].dt.minute/60 + df['date'].dt.second/3600 - 12)

    df['user_statuses_count'] = df['user_statuses_count'].astype(int)
    df['user_verified'] = df['user_verified'].astype(int)
    df['date'] =pd.to_datetime(df['date'])
    df['hashtags'] = df['hashtags'].astype(str).str.lower()
    df['urls'] = df['urls'].astype(str).str.lower()
    df['user_mentions'] = df['user_mentions'].astype(str).str.lower()
    df['text'] = df['text'].astype(str).str.lower()
    df['text_length'] = df['text'].apply(lambda x: len(x))

    #remove URL
    df['text_proc'] = df['text'].str.replace(r'http(\S)+', r'')
    df['text_proc'] = df['text_proc'].str.replace(r'http ...', r'')
    df['text_proc'] = df['text_proc'].str.replace(r'http', r'')
    df[df['text_proc'].str.contains(r'http')]
    # remove RT, @
    df['text_proc'] = df['text_proc'].str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')
    df[df['text_proc'].str.contains(r'RT[ ]?@')]
    df['text_proc'] = df['text_proc'].str.replace(r'@[\S]+', r'')
    #remove non-ascii words and characters
    df['text_proc'] = [''.join([i if ord(i) < 128 else '' for i in text]) for text in df['text_proc']]
    df['text_proc'] = df['text_proc'].str.replace(r'_[\S]?',r'')
    #remove &, < and >
    df['text_proc'] = df['text_proc'].str.replace(r'&amp;?',r'and')
    df['text_proc'] = df['text_proc'].str.replace(r'&lt;',r'<')
    df['text_proc'] = df['text_proc'].str.replace(r'&gt;',r'>')
    # remove extra space
    df['text_proc'] = df['text_proc'].str.replace(r'[ ]{2, }',r' ')
    # insert space between punctuation marks
    df['text_proc'] = df['text_proc'].str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    df['text_proc'] = df['text_proc'].str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
    # strip white spaces at both ends
    df['text_proc'] = df['text_proc'].str.strip()



def token(df,feature,saving_path = None, N_WORDS = N_WORDS, merge = False):

    """
    Token, is a function that uses dictionnaries created from the columns we want to change to clusterize the tweets
    It creates the variables mentioned for Preproc 1 in the report

    Inputs: 
        feature : str, 
            Columns from which to create labels
        saving_path : str, default = None
            The path where to save the output DataFrame
        N_WORDS : int, default = constants.N_WORDS
            Number of words stored into the dictionnary
        merge : boolean, default = False.
            If merge is True, merge the dummy DataFrame with the original one. 
    
    Output: 
        None : A label column is added to the self.table if merge == False, otherwise the dummy DataFrame is merged to 
        self.table and stored in saving_path.

    """

    N_SAMPLE = len(df)

    if feature == urls :
        df[urls] = df[urls].apply(lambda x : remove_char(x))


    #We tokenize the feature for each tweet, then pad the output of the tokenization
    tokenizer = Tokenizer(num_words = N_WORDS)
    tokenizer.fit_on_texts(df[feature])
    sequences = tokenizer.texts_to_sequences(df[feature])
    padded = pad_sequences(sequences)
    dummy_texts = np.zeros(shape = (N_SAMPLE,N_WORDS))
    for i in range(padded.shape[0]) : 
        for j in range(padded.shape[1]):
            if padded[i,j]!=0:
                dummy_texts[i,padded[i,j]]=1
    
    b = np.sum(dummy_texts,axis=1)

    for i in range(dummy_texts.shape[0]):
        dummy_texts[i,:] /= b[i]
    dummy_texts = np.nan_to_num(dummy_texts)
    dummy_texts = dummy_texts.astype(np.float32)


    #We use PCA on the padded matrix
    pca = PCA(20)
    X_new = pca.fit_transform(dummy_texts)

    #We remove the feature from the dataset
    df.drop([feature],axis=1)

    if merge : 
        df_x = pd.DataFrame(X_new)
        df_x[ID] = np.arange(N_SAMPLE)
        batch_merger(df,df_x,saving_path,ID,ID)
        df = pd.read_csv(saving_path)
    
    #For all other features, we clusterize using the PCA coordinates we had computed
    else : 
        clf = KMeans(cluster_dict[feature])
        labels = clf.fit_predict(X_new)
        df[labels_dict[feature]] = labels



def table_columns_removal(df):
    """
    Removes all useless features from the dataset
    """
    df.drop(columns = ['id', 'timestamp', 'date', 'hashtags', 'text', 'text_proc', 'user_mentions', 'urls', 'text_labels'], axis = 1, inplace = True)



def table_save(df, saving_path):
    df.to_csv(saving_path, index = False)


#We load BERT Base
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

def embed(token) :
    """
    Returns the embedding of a tokenized and padded sequence of words
    """
    return model(**token).last_hidden_state[:,0,:].detach().numpy()[0]



if __name__ == "__main__" :

    for d in ['evaluation.csv', 'train.csv'] :

        #We load the data 
        data = pd.read_csv('data/' + d)

        #Then we do some basic preprocessing
        columns_preprocessing(data)

        #We create Preproc 1's features
        for feature in cluster_dict.keys() :
            token(data, feature)

        #We tokenize the dataset
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized = data['text_proc'].apply((lambda x: tokenizer(x, padding = True, return_tensors = 'pt')))
        
        #Then we compute BERT embeddings, using parallelization
        with cf.ProcessPoolExecutor() as executor :
            embedded = list(executor.map(embed, tokenized))

        embedded = np.array(embedded)

        #We use PCA to reduce the dimensionality of the BERT embeddings
        pca = PCA(50)
        embedded_pca = pca.fit_transform(embedded)
        emb_df = pd.DataFrame(embedded_pca)

        data = pd.concat([data, emb_df], axis=1)

        #We remove all useless columns
        table_columns_removal(data)

        #We save the data
        data.to_csv('data/preprocessed_' + d, index = False)