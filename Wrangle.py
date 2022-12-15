import pandas as pd
import numpy as np
from env import host, username, password
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import seaborn as sns
import matplotlib.pyplot as plt

def get_games():
    '''this function will return my csv into my notebook'''
    return pd.read_csv('game_info.csv')


def prep_data(df):
    ''' This Function will prep my data for input into my model'''
    #Here I am getting rid of rowsw where metacritic is null as this is where I am deriving my target from and need all things being tested to have a metacritic review.
    df = df[df.metacritic.isnull() == False]
    df = df[df.genres.isna() == False]
    #creating target
    df['metacritic_good_game'] = df.metacritic[df.metacritic > 90 ]
    df.metacritic_good_game[df.metacritic_good_game > 0] = 1
    df.metacritic_good_game = df.metacritic_good_game.fillna(0)
    #PUBLISHER ENCODING:
    Publisher_list = ['']
    word = ''
    for x in df.publishers[df.metacritic > 90].tolist():
        for a in str(x):
            if a != '|':
                word = word + a
            else:
                if word not in Publisher_list:
                    Publisher_list.append(word)
                    word = ''
                word = ''
        if word not in Publisher_list:
            Publisher_list.append(word)
            word = ''
        word = ''
    for x in Publisher_list:
        Publisher_list.remove(x)
        x = x.replace(' ','_')
        Publisher_list.append(x)
    Publisher_list.remove('')
    Publisher_list.remove('nan')
    for x in Publisher_list:
        df[f'Publisher_{x}'] = df.publishers.str.contains(x)
    #PLATFORM ENCODING:
    platform_list = ['PC','Xbox One']
    word = ''
    for x in df.platforms.tolist():
        for a in str(x):
            if a != '|':
                word = word + a
            else:
                if word not in platform_list:
                    platform_list.append(word)
                    word = ''
                word = ''
        if word not in platform_list:
            platform_list.append(word)
            word = ''
        word = ''
    platform_list.remove('')
    platform_list.remove('nan')
    for x in platform_list:
        platform_list.remove(x)
        x = x.replace(' ','_')
        platform_list.append(x)
    for x in platform_list:
        df[f'Platform_{x}'] = df.platforms.str.contains(x)
    #GENRE ENCODING:
    ### FINDS UNIQUE WORDS FROM LIST OF GENRES
    genre_list = ['Action','Adventure']
    word = ''
    for x in df.genres.tolist():
        for a in str(x):
            if a != '|':
                word = word + a
            else:
                if word not in genre_list:
                    genre_list.append(word)
                    word = ''
                word = ''
        if word not in genre_list:
            genre_list.append(word)
            word = ''
        word = ''
    genre_list.remove('')
    for x in genre_list:
        genre_list.remove(x)
        x = x.replace(' ','_')
        genre_list.append(x)
    for x in genre_list:
        df[f'Genre_{x}'] = df.genres.str.contains(x)
    #NO WHITESPACE IN COLUMN NAMES:
    df.columns = df.columns.str.replace(' ','_')
     #drops
    df = df.drop(columns = ['rating','rating_top','ratings_count','suggestions_count','reviews_count','added_status_owned','added_status_beaten','added_status_dropped','added_status_playing','esrb_rating','website','updated','tba','name','slug','id','developers','genres','platforms','publishers','released'])
    df = df.dropna()

    return df

def split_games(df):
    train, validate, test = my_train_test_split(df, 'metacritic_good_game')
    x_train = train.drop(columns = 'metacritic_good_game')
    y_train = train.metacritic_good_game
    x_validate = validate.drop(columns = 'metacritic_good_game')
    y_validate = validate.metacritic_good_game
    x_test = test.drop(columns = 'metacritic_good_game')
    y_test = test.metacritic_good_game
    return train,validate,test, x_train, y_train, x_validate, y_validate, x_test, y_test
    #below is the function that splits the data into a specified size
def my_train_test_split(df, target):

    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])

    return train, validate, test

def Scale_data(x_train,x_validate,x_test):
    #SCALING ALL NUMERIC DATA
    list = []
    for x in x_train.columns:
        list.append(x)
    x_train[list] = scaler.fit_transform(x_train[list])
    x_validate[list] = scaler.fit_transform(x_validate[list])
    x_test[list] = scaler.fit_transform(x_test[list])




