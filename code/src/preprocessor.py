#####################################
############ Preprocessing ##########
#####################################

# This python file is dedicated to Preprocessing

from src.utils import *
from src.constants import *
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


class preprocessor: 

    def __init__(self,table):

        self.table = table
        self.N_SAMPLE = len(self.table)
        
    

    def columns_preprocessing(self):

        """
        Basic preprocessing the columns. 

        We create the day, time, text_length, hashtags_length and user_mentions_lenght variables.
        Bool columns are changed into 1 and 0. 
        String columns are made into lower case, nan are changed into strings, non-letter characters are removed.
        """

        self.table[date] = pd.to_datetime(self.table['timestamp'],unit='ms')
        self.table[day] = self.table[date].dt.dayofweek.astype('category')
        self.table[time] = np.abs(self.table[date].dt.hour + self.table[date].dt.minute/60 + self.table[date].dt.second/3600 - 12)

        self.table[user_total_tweet] = self.table[user_total_tweet].astype(int)
        self.table[user_verified] = self.table[user_verified].astype(int)
        self.table[date] =pd.to_datetime(self.table[date])
        self.table[hashtags] = self.table[hashtags].astype(str).str.lower()
        self.table[urls] = self.table[urls].astype(str).str.lower()
        self.table[user_mentions] = self.table[user_mentions].astype(str).str.lower()
        self.table[hasthags_length] = self.table[hashtags].apply(lambda x : hstg_len(x))
        self.table[user_mentions_lenght] = self.table[user_mentions].apply(lambda x : hstg_len(x))


        self.table[text] = self.table[text].str.replace(r'http(\S)+', r'')
        self.table[text] = self.table[text].str.replace(r'http ...', r'')
        self.table[text] = self.table[text].str.replace(r'http', r'')
        self.table[self.table[text].str.contains(r'http')]
        # remove RT, @
        self.table[text] = self.table[text].str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')
        self.table[text] = self.table[text].str.replace(r'@[\S]+', r'')
        #remove non-ascii words and characters
        self.table[text] = [''.join([i if ord(i) < 128 else '' for i in text]) for text in self.table[text]]
        self.table[text] = self.table[text].str.replace(r'_[\S]?',r'')
        #remove &, < and >
        self.table[text] = self.table[text].str.replace(r'&amp;?',r'and')
        self.table[text] = self.table[text].str.replace(r'&lt;',r'<')
        self.table[text] = self.table[text].str.replace(r'&gt;',r'>')
        # remove extra space
        self.table[text] = self.table[text].str.replace(r'[ ]{2, }',r' ')
        # insert space between punctuation marks
        self.table[text] = self.table[text].str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
        self.table[text] = self.table[text].str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
        # strip white spaces at both ends
        self.table[text] = self.table[text].str.strip()        
        self.table[text] = self.table[text].astype(str).str.lower()
        self.table[text_length] = self.table[text].apply(lambda x: len(x))

    def token(self,feature,saving_path = None, N_WORDS = N_WORDS, merge = True):

        """
        Token, is a function that uses dictionnaries created from the columns we want to change to clusterize the tweets
        It creates the variables mentioned for Preproc 1 and Preproc 1bis in the report

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

        vectorizer = TfidfVectorizer()

        if feature == urls :
            self.table[urls] = self.table[urls].apply(lambda x : remove_char(x))
        
        #We tokenize the feature for each tweet, then pad the output of the tokenization
        tokenizer = Tokenizer(num_words = N_WORDS)
        tokenizer.fit_on_texts(self.table[feature])
        sequences = tokenizer.texts_to_sequences(self.table[feature])
        padded = pad_sequences(sequences)
        dummy_texts = np.zeros(shape = (self.N_SAMPLE,N_WORDS))
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

        
        #We compute the TF-IDF matrix then apply NMF only for text
        if feature == text and merge: 

            X = vectorizer.fit_transform(self.table[feature])
            nmf = NMF(n_components=10 ,random_state=1,
            alpha=.1, l1_ratio=.5).fit_transform(X)
            X_df =  pd.DataFrame(nmf)
            X_df[ID] = self.table[ID]                   
            columns = list(self.table.columns)        
            X_col = list(X_df.columns)
            for i in range(len(X_col)-1):
                columns.append(X_col[i])
            batch_merger(self.table,X_df,saving_path,ID,ID,columns)
            self.table = pd.read_csv(saving_path)
            X_new_df = pd.DataFrame(X_new[:,:5])
            X_new_df[ID] = self.table[ID] 
            X_col = list(X_new_df.columns)
            for i in range(len(X_col)-1):
                columns.append(X_col[i])
            batch_merger(self.table,X_new_df,saving_path,ID,ID,columns)
            self.table = pd.read_csv(saving_path)
            
        #For all other features, we clusterize using the PCA coordinates we had computed
        else : 
            clf = KMeans(cluster_dict[feature])
            labels = clf.fit_predict(X_new)
            self.table[labels_dict[feature]] = labels
        
        #We remove the feature from the dataset
        self.table.drop([feature],axis=1, inplace = True)


    def table_columns_removal(self):
        """
        Removes the date column we had created previously to get time and day
        """
        try:
            self.table.drop(columns = [date], axis = 1, inplace = True)
        except:
            print(self.table.columns)
   
    
    def table_save(self, saving_path):
        self.table.to_csv(saving_path, index = False)


    def table_split(self):
        self.X_table = self.table.loc[:, self.table.columns != retweet_count]
        self.Y_table = self.table[retweet_count]


    def run(self, saving_path = None):
        """
        Run this function to preprocess the table.
        """
        self.columns_preprocessing()

        # We select the features we want to clusterize then we clusterize them
        for feature in cluster_dict.keys():
            print(feature)
            self.token(feature,saving_path)
        
        self.table_columns_removal()
        if saving_path != None : 
            self.table_save(saving_path)


    


