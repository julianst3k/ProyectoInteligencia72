
import numpy as np

import FATS

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import pandas as pd


## Leo el archivo
data = pd.read_pickle('periodic_dataset/periodic_detections.pkl')

data.drop_duplicates()

dataValues = data.index.unique()

labels = pd.read_pickle('periodic_dataset/periodic_labels.pkl')

preproc = labels.groupby('classALeRCE').count()


values = preproc.index.unique()

minimumTrainingSet = preproc.min().iloc[0]

# Hago 80%-20% para el training set, en base al minimo necesario, el cual es minimumTrainingSet
ultimate_label_train = pd.DataFrame()
ultimate_label_test = pd.DataFrame()

for j in (values):
    chosen_ids = np.random.choice(int(preproc.loc[j,'period']), replace=False, size=minimumTrainingSet)
    chosen_ids_training = chosen_ids[:int(len(chosen_ids)*0.8)]
    chosen_ids_testing = chosen_ids[int(len(chosen_ids)*0.8):]
    final_label_aux = labels.loc[labels['classALeRCE']==j]
    final_label_training = final_label_aux.iloc[chosen_ids_training]
    final_label_testing = final_label_aux.iloc[chosen_ids_testing]


    final_data = data.iloc[chosen_ids]
    #print(final_label_aux)
    ultimate_label_train = pd.concat([ultimate_label_train,final_label_training])
    ultimate_label_test = pd.concat([ultimate_label_test, final_label_testing])

print(ultimate_label_train)

## Tengo divididos los labels, puedo aplicar FATS a todos los elementos escogidos de la muestra
#testSet =

#trainingSet =



feature_list = [
    'Amplitude', 'AndersonDarling', 'Autocor_length',
    'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
    'Con', 'Eta_e', 'Gskew', 'MaxSlope', 'Mean',
    'Meanvariance', 'MedianAbsDev', 'MedianBRP',
    'PairSlopeTrend', 'PercentAmplitude', 'Q31',
    'Rcs', 'Skew', 'SmallKurtosis',
    'Std', 'StetsonK']

oids = data.index.unique()

fats_fs = FATS.FeatureSpace(
    Data=['magnitude', 'time', 'error'],
    featureList=feature_list,
    excludeList=[
        'StetsonJ',
        'StetsonL',
        'Eta_color',
        'Q31_color',
        'Color'])
ultimate_data_test = []
ultimate_data_train = []

stars_training = ultimate_label_train.index.unique()
print(stars_training)
print(ultimate_label_train)
def generateFATS(label):
    stars_training = label.index.unique()
    for star in stars_training:
        first_lc = data.loc[star]
        first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
        flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
        flc2 = first_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
        valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
        if(len(valores)<5):
            index = label.loc[star]
            label.drop(star, axis=0, inplace=True)
        else:
            features1 = fats_fs.calculateFeature(valores.T).result().tolist()
            ultimate_data_train.append(features1)
#stars_testing = ultimate_label_test.index.unique()

#for star in stars_testing:
#    first_lc = data.loc[star]
#    first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
#    flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
#    flc2 = first_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
#    valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
#    if(len(valores)<5):
#        index = ultimate_label_test.loc[star]
#        ultimate_label_test.drop(star, axis=0, inplace=True)
#    else:
#        features1 = fats_fs.calculateFeature(valores.T).result().tolist()
#        ultimate_data_test.append(features1)

num_steps = 100

num_classes = 6

num_features = 23

num_trees = 10

max_nodes = 1000

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Aqui saco los valores de los labels

values_train = ultimate_label_train['classALeRCE'].values
ultimate_values_train = []
for j in values_final:


print(values_final)
