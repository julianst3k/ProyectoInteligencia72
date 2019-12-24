
import numpy as np

from randomForest import *

import math

import FATS

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt

import os

import pickle

from sklearn.utils.multiclass import unique_labels

feature_list = [
    'Amplitude', 'AndersonDarling', 'Autocor_length',
    'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
    'Con', 'Eta_e', 'Gskew', 'MaxSlope', 'Mean',
    'Meanvariance', 'MedianAbsDev', 'MedianBRP',
    'PairSlopeTrend', 'PercentAmplitude', 'Q31',
    'Rcs', 'Skew', 'SmallKurtosis',
    'Std', 'StetsonK']
exclude_list = ['StetsonJ',
                'StetsonL',
                'Eta_color',
                'Q31_color',
                'Color']
# Para activar el file
toFile = True
# Para sacar la media entre dos modelos
means = None
# Para usar dos bandas
both = True
# Para usar un modelo
one_band = False
# Para usar la tercera solución, aka dos modelos
thirdSol = False
if toFile:
    data = openDataFATSGenerado("fats_processed_final.pkl", means=means, both=both, thirdSol = thirdSol)
else:
    data = generateDataFATS(toFile=False, means=means)
## Por algún motivo existe un NaN a la hora de generar resultados, por lo cual se busco por aca. Es de suponer que
## Dado que ocurrió una vez puede ocurrir mas veces, esto es provisorio y por ende se generara un detector de nan
## Dsps
nansuspect = []
for a in data.ultimate_data_train_band2:
    for j in a:
        if(math.isnan(j)):
            nansuspect.extend(a)
if(len(nansuspect)>0):
    index = data.ultimate_data_train_band2.index(nansuspect)
    data.ultimate_data_train_band2.pop(index)
    data.ultimate_data_train_band1.pop(index)
    if both:
        data.ultimate_data_train.pop(index)
    stars_training = data.ultimate_label_train.index.unique()
    starnan = stars_training[index]
    data.ultimate_label_train.drop(starnan, axis=0, inplace=True)
if thirdSol:
    data.fixColor()
firstClassifier = classifier()
data.ultimateDataLength()
firstClassifier.classifier(data, means=means, thirdSol=thirdSol, one_band=one_band)
feature_list.append('Period')
feature_list2 = [
    'Amplitude2', 'AndersonDarling2', 'Autocor_length2',
    'Beyond1Std2', 'CAR_sigma2', 'CAR_mean2', 'CAR_tau2',
    'Con2', 'Eta_e2', 'Gskew2', 'MaxSlope2', 'Mean2',
    'Meanvariance2', 'MedianAbsDev2', 'MedianBRP2',
    'PairSlopeTrend2', 'PercentAmplitude2', 'Q312',
    'Rcs2', 'Skew2', 'SmallKurtosis2',
    'Std2', 'StetsonK2','Color']
feature_list.extend(feature_list2)
firstClassifier.importanceVariable(feature_list)
#Para sacar el .dot del árbol y la importancia de las variables, eliminar los #
print("Solo banda 1: "+ str(firstClassifier.oneband)+" Solo banda 2: "+str(firstClassifier.twoband)+" Ambas: "+str(firstClassifier.mix))
plt.show()