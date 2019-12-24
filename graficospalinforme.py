
import numpy as np
import plotly.graph_objects as go
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

input = open("fats_processed_final.pkl", 'rb')
data = pickle.load(input)
length = len(data.ultimate_data_test[0])
values_test_final = data.ultimate_label_test['classALeRCE'].values
onehot_values_test = []
for j in values_test_final:
    cyka = [0,0,0,0,0,0]
    cyka[np.where(data.values == j)[0][0]] = 1
    onehot_values_test.append(np.where(data.values==j)[0][0])
color = [row[17] for row in data.ultimate_data_test]
c = ['#99ccff','#cc3300','#009933','#ffff00','#666699','#669999']
colors = [[],[],[],[],[],[]]
for i in range(len(color)):
    colors[onehot_values_test[i]].append(color[i])

fig = go.Figure()

for val in range(6):
    fig.add_trace(go.Box(y=colors[val], name=data.values[val],
                marker_color = c[val]))
fig.update_layout(
    title='Distribución de color según clase',
    yaxis_title="Desviación estándar",
    xaxis_title="Clases")
fig.show()
