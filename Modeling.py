import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier,plot_tree,export_text
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def get_baseline(train):
    baseline_accuracy = round((train.metacritic_good_game == 0).mean(), 4) * 100
    return(baseline_accuracy)

def get_DTC(x_train,y_train,x_validate,y_validate,depth):
    DTC = DecisionTreeClassifier(max_depth = depth, random_state = 123)
    DTC.fit(x_train,y_train)
    y_preds_train = pd.DataFrame({
    'y_act': y_train,
    'baseline': 0,
    'DTC_train': DTC.predict(x_train)})
    y_preds_validate = pd.DataFrame({
    'y_act': y_validate,
    'baseline_': 0,
    'DTC_validate': DTC.predict(x_validate)})
    print(pd.DataFrame(classification_report(y_preds_validate.y_act,y_preds_validate.DTC_validate, output_dict=True)))
                     
def get_RF(x_train,y_train,x_validate,y_validate,depth,samples):
    rf = RandomForestClassifier(max_depth = depth, min_samples_leaf = samples,random_state = 123)
    rf.fit(x_train,y_train)
    y_preds_train = pd.DataFrame({
    'y_act': y_train,
    'baseline': 97,
    'rf': rf.predict(x_train)})
    y_preds_validate = pd.DataFrame({
    'y_act': y_validate,
    'baseline': 97,
    'rf': rf.predict(x_validate)})
    print(pd.DataFrame(classification_report(y_preds_validate.y_act, y_preds_validate.rf, output_dict=True)))
    
def get_LOGREG(x_train,y_train,x_validate,y_validate,c):
    logreg = LogisticRegression(C = c,random_state = 123)
    logreg.fit(x_train,y_train)
    y_preds_train = pd.DataFrame({
    'y_act': y_train,
    'baseline': 97,
    'logreg': logreg.predict(x_train)})
    y_preds_validate = pd.DataFrame({
    'y_act': y_validate,
    'baseline': 97,
    'logreg': logreg.predict(x_validate)})
    print(pd.DataFrame(classification_report(y_preds_validate.y_act, y_preds_validate.logreg, output_dict=True)))
    
def test_DTC(x_train,y_train,x_test,y_test,depth):
    DTC = DecisionTreeClassifier(max_depth = depth, random_state = 123)
    DTC.fit(x_train,y_train)
    y_preds = pd.DataFrame({
    'y_act': y_test,
    'baseline': 97,
    'DTC': DTC.predict(x_test)})
    print(pd.DataFrame(classification_report(y_preds.y_act,y_preds.DTC,output_dict=True)))
    