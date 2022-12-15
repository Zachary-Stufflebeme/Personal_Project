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
from scipy import stats

def get_shooter_viz(train):
    " get graph of game rating for upsets and non-upsets"
    # assign values and labels
    values = [train.metacritic_good_game[(train.Genre_Shooter == True)].mean(),train.metacritic_good_game.mean()]
    labels = ['Shooters','All']
    # generate and display graph
    plt.bar(height=values, x=labels, color=['#FFC3A0', '#C0D6E4'])
    plt.ylim(0,.2)
    plt.title('The good game avg between Shooter games and all games')
    plt.show()

def get_blizzard_viz(train):
    " get graph of game rating for upsets and non-upsets"
    # assign values and labels
    values = [train.metacritic_good_game[(train.Publisher_Blizzard_Entertainment == True)].mean(),train.metacritic_good_game.mean()]
    labels = ['Blizzard','All']
    # generate and display graph
    plt.bar(height=values, x=labels, color=['#FFC3A0', '#C0D6E4'])
    plt.ylim(0,1)
    plt.title('The good game avg between blizzard games and all games')
    plt.show()

def get_indie_viz(train):
    " get graph of game rating for upsets and non-upsets"
    # assign values and labels
    values = [train.metacritic_good_game[(train.Genre_Indie == True)].mean(),train.metacritic_good_game.mean()]
    labels = ['Indie','All']
    # generate and display graph
    plt.bar(height=values, x=labels, color=['#FFC3A0', '#C0D6E4'])
    plt.ylim(0,.2)
    plt.title('The good game avg between Indie games and all games')
    plt.show()

def get_PC_viz(train):
    " get graph of game rating for upsets and non-upsets"
    # assign values and labels
    values = [train.metacritic_good_game[(train.Platform_PC == True)].mean(),train.metacritic_good_game.mean()]
    labels = ['PC','All']
    # generate and display graph
    plt.bar(height = values, x = labels, color =['#FFC3A0','#C0D6E4'])
    plt.ylim(0,.2)
    plt.title('The good game average between PC games and all games')
    plt.show()

def get_RPG_viz(train):
    " get graph of game rating for upsets and non-upsets"
    # assign values and labels
    values = [train.metacritic_good_game[(train.Genre_RPG == True)].mean(),train.metacritic_good_game.mean()]
    labels = ['RPG','All']
    # generate and display graph
    plt.bar(height = values, x = labels, color =['#FFC3A0','#C0D6E4'])
    plt.ylim(0,.2)
    plt.title('The good game average between RPG games and all games')
    plt.show()


def ttest(train):
    alpha = .05
    good_game =  train [ train.metacritic_good_game == 1]
    bad_game = train [ train.metacritic_good_game == 0]
    tstat, p = stats.ttest_ind(good_game.suggestions_count,
                               bad_game.suggestions_count,
                               equal_var=False
                               )
    print(f'The p-value is less than the alpha: {p < alpha}')
#%%
def chi_square(train, feature):
    alpha = .05
    observed = pd.crosstab(train[feature], train.metacritic_good_game)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'The p-value is less than the alpha: {p < alpha}')

