###################################################################################
#This python script is in order to define the constant we will use in the project #
###################################################################################


# Column names 

ID                   = 'id'
date                 = 'timestamp'
retweet_count        = 'retweet_count'
user_verified        = 'user_verified'
user_total_tweet     = 'user_statuses_count'
user_total_followers = 'user_followers_count'
user_total_friends   = 'user_friends_count'
user_mentions        = 'user_mentions'
urls                 = 'urls'
hashtags             = 'hashtags'
text                 = 'text'
text_length          = 'text_length'
day                  = 'day'
time                 = 'time'
hasthags_length      = 'hasthags_length'
user_mentions_lenght = 'user_mentions_lenght'
# Label names

mention_labels = 'mention_labels'
hashtag_labels = 'hashtag_labels'
url_labels     = 'url_labels'
text_labels    = 'text_labels'


# Global Variables 

N_WORDS = 1000 

# Cluster number 

N_URLS_CLUSTERS = 3 

# There is 3 clusters here :
# - No link
# - Twitter links
# - Other links

N_HTGS_CLUSTERS = 3

# There is 4 clusters here : 
# - No #
# - COVID-19 #
# - coronavirus #
# - Other #

# N_TWEET_CLUSTERS = 2
N_TWEET_CLUSTERS = 4

# There is 2 clusters


N_MENTION_CLUSTERS = 3

# There is 3 clusters : 
# - No mention 
# - Casual user mention 
# - Donald Trump mention

cluster_dict = {
    user_mentions:N_MENTION_CLUSTERS,
    urls:N_URLS_CLUSTERS,
    hashtags:N_HTGS_CLUSTERS,
    text:N_TWEET_CLUSTERS
}

labels_dict = {
    user_mentions:mention_labels,
    urls:         url_labels,
    hashtags:     hashtag_labels,
    text:         text_labels
}

non_scalable_col = [retweet_count,mention_labels,url_labels,hashtag_labels,text_labels]