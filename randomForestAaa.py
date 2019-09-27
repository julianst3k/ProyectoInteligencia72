
import numpy as np

import FATS

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels

## Leo el archivo
data = pd.read_pickle('periodic_dataset/periodic_detections.pkl')

data.drop_duplicates()

dataValues = data.index.unique()

labels = pd.read_pickle('periodic_dataset/periodic_labels.pkl')
labels.dropna()

preproc = labels.groupby('classALeRCE').count()

values = preproc.index.unique()


minimumTrainingSet = preproc.min().iloc[0]

# Hago 80%-20% para el training set, en base al minimo necesario, el cual es minimumTrainingSet
ultimate_label_train = pd.DataFrame()
ultimate_label_test = pd.DataFrame()
weight = []

for j in (values):
    val = int(preproc.loc[j, 'period'])
    if(int(preproc.loc[j,'period'])>1000):
        val = 1000
    chosen_ids = np.random.choice(int(preproc.loc[j,'period']), replace=False, size=val)
    weight.append(int(preproc.loc[j,'period']))
    print(len(chosen_ids))
    chosen_ids_training = chosen_ids[:int(len(chosen_ids)*0.8)]
    chosen_ids_testing = chosen_ids[int(len(chosen_ids)*0.8):]
    final_label_aux = labels.loc[labels['classALeRCE']==j]
    final_label_training = final_label_aux.iloc[chosen_ids_training]
    final_label_testing = final_label_aux.iloc[chosen_ids_testing]
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
i=0
for star in stars_training:
    i+=1
    first_lc = data.loc[star]
    first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
    flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
    flc1.dropna()
    flc2 = first_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
    valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
    if(len(valores)<10 or ultimate_label_train.loc[star, 'period']=="\\\\nodata"):
        index = ultimate_label_train.loc[star]
        ultimate_label_train.drop(star, axis=0, inplace=True)
    else:
        features1 = fats_fs.calculateFeature(valores.T).result().tolist()
        features1.append(float(ultimate_label_train.loc[star, 'period']))
        ultimate_data_train.append(features1)
    if(i%100==0):
        print(i)
print("xd")
stars_testing = ultimate_label_test.index.unique()
for star in stars_testing:
    first_lc = data.loc[star]
    first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
    flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
    flc2 = first_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
    flc1.dropna()
    valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values

    if(len(valores)<10 or ultimate_label_test.loc[star, 'period']=="\\\\nodata"):
        index = ultimate_label_test.loc[star]
        ultimate_label_test.drop(star, axis=0, inplace=True)
    else:
        features1 = fats_fs.calculateFeature(valores.T).result().tolist()
        features1.append(float(ultimate_label_test.loc[star, 'period']))
        ultimate_data_test.append(features1)
print("xD")
num_steps = 100

num_classes = 6

num_features = 23

num_trees = 10

max_nodes = 1000

rf = RandomForestClassifier(n_estimators = 50, n_jobs=2, random_state = 10, class_weight="balanced")
# Aqui saco los valores de los labels

values_train_final = ultimate_label_train['classALeRCE'].values
onehot_values_train = []
for j in values_train_final:
    cyka = [0,0,0,0,0,0]
    cyka[5-np.where(values == j)[0][0]] = 1
    onehot_values_train.append(cyka)
p=np.random.permutation(len(onehot_values_train))
ultimate_data_train = np.array(ultimate_data_train)[p]
onehot_values_train = np.array(onehot_values_train)[p]
values_test_final = ultimate_label_test['classALeRCE'].values

onehot_values_test = []
for j in values_test_final:
    cyka = [0,0,0,0,0,0]
    cyka[5-np.where(values == j)[0][0]] = 1
    onehot_values_test.append(cyka)
rf.fit(ultimate_data_train, onehot_values_train)
predictions = rf.predict(ultimate_data_test)
# Calculate the absolute errors
errors = abs(predictions - onehot_values_test)
print(errors)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
onehot_values_test = np.array(onehot_values_test).argmax(1)
predictionLol = predictions
predictions = np.array(predictions).argmax(1)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title="Matriz de confusión normalizada",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(onehot_values_test, predictions, values)
plt.show()