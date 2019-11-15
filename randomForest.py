
import numpy as np

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

## En esta implementación refracte el código en términos de clases, pero no fue mi implementación inicial,
## por lo cual se cambiará a futuro de todos modos
class DataProcessor:
    ## Inicialización de los datos
    def __init__(self):
        data = pd.read_pickle('periodic_dataset/periodic_detections.pkl')
        data.drop_duplicates()
        self.data = data
        self.dataValues = data.index.unique()
        self.labels = pd.read_pickle('periodic_dataset/periodic_labels.pkl')
        self.labels.dropna()
    # Se genera la partición
    def generateLabelsPartition(self, max_data = 1000):
        preproc = self.labels.groupby('classALeRCE').count()
        self.values = preproc.index.unique()
        self.ultimate_label_train = pd.DataFrame()
        self.ultimate_label_test = pd.DataFrame()
        weight = []
        for j in (self.values):
            val = int(preproc.loc[j, 'period'])
            if(val>max_data):
                 val = 1000
            chosen_ids = np.random.choice(int(preproc.loc[j,'period']), replace=False, size=val)
            weight.append(int(preproc.loc[j,'period']))
            chosen_ids_training = chosen_ids[:int(len(chosen_ids)*0.8)]
            chosen_ids_testing = chosen_ids[int(len(chosen_ids)*0.8):]
            final_label_aux = self.labels.loc[self.labels['classALeRCE']==j]
            final_label_training = final_label_aux.iloc[chosen_ids_training]
            final_label_testing = final_label_aux.iloc[chosen_ids_testing]
            self.ultimate_label_train = pd.concat([self.ultimate_label_train,final_label_training])
            self.ultimate_label_test = pd.concat([self.ultimate_label_test, final_label_testing])
    def applyFATS(self, feature_list, exclude_list):
        fats_fs = FATS.FeatureSpace(
            Data=['magnitude', 'time', 'error'],
            featureList=feature_list,
            excludeList=exclude_list)
        oids = self.data.index.unique()
        self.ultimate_data_test_mixed = []
        self.ultimate_data_train_mixed = []
        stars_training = self.ultimate_label_train.index.unique()
        for star in stars_training:
            first_lc = self.data.loc[star]
            first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
            second_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc2 = second_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
            flc1.dropna()
            flc2.dropna()
            valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            valores2 = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            if (len(valores) < 8 or len(valores2) < 8 or self.ultimate_label_train.loc[star, 'period'] == "\\\\nodata"):
                index = self.ultimate_label_train.loc[star]
                self.ultimate_label_train.drop(star, axis=0, inplace=True)
            else:
                features1 = fats_fs.calculateFeature(valores.T).result().tolist()
                features2 = fats_fs.calculateFeature(valores2.T).result().tolist()
                features1.append(float(self.ultimate_label_train.loc[star, 'period']))
                features1.extend(features2)
                features1.append(features1[11] - features2[11])
                self.ultimate_data_train_mixed.append(features1)
        stars_testing = self.ultimate_label_test.index.unique()
        for star in stars_testing:
            first_lc = self.data.loc[star]
            first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
            second_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc2 = second_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
            flc1.dropna()
            flc2.dropna()
            valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            valores2 = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            if (len(valores) < 8 or len(valores2) < 8 or self.ultimate_label_test.loc[star, 'period'] == "\\\\nodata"):
                index = self.ultimate_label_test.loc[star]
                self.ultimate_label_test.drop(star, axis=0, inplace=True)
            else:
                features1 = fats_fs.calculateFeature(valores.T).result().tolist()
                features2 = fats_fs.calculateFeature(valores2.T).result().tolist()
                features1.append(float(self.ultimate_label_test.loc[star, 'period']))
                features1.extend(features2)
                features1.append(features1[11] - features2[11])
                self.ultimate_data_test_mixed.append(features1)

    def applyFATSTwoLists(self, feature_list, exclude_list):
        fats_fs = FATS.FeatureSpace(
            Data=['magnitude', 'time', 'error'],
            featureList=feature_list,
            excludeList=exclude_list)
        oids = self.data.index.unique()
        self.ultimate_data_test_band1 = []
        self.ultimate_data_test_band2 = []
        self.ultimate_data_train_band1 = []
        self.ultimate_data_train_band2 = []
        stars_training = self.ultimate_label_train.index.unique()
        for star in stars_training:
            first_lc = self.data.loc[star]
            first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
            second_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc2 = second_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
            flc1.dropna()
            flc2.dropna()
            valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            valores2 = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            if (len(valores) < 8 or len(valores2) < 8 or self.ultimate_label_train.loc[star, 'period'] == "\\\\nodata"):
                index = self.ultimate_label_train.loc[star]
                self.ultimate_label_train.drop(star, axis=0, inplace=True)
            else:
                features1 = fats_fs.calculateFeature(valores.T).result().tolist()
                features2 = fats_fs.calculateFeature(valores2.T).result().tolist()
                features1.append(float(self.ultimate_label_train.loc[star, 'period']))
                self.ultimate_data_train_band1.append(features1.copy())
                features_band2 = features2.copy()
                features_band2.append(float(self.ultimate_label_train.loc[star, 'period']))
                self.ultimate_data_train_band2.append(features_band2.copy())
        stars_testing = self.ultimate_label_test.index.unique()
        for star in stars_testing:
            first_lc = self.data.loc[star]
            first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
            second_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
            flc2 = second_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
            flc1.dropna()
            flc2.dropna()
            valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            valores2 = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
            if (len(valores) < 8 or len(valores2) < 8 or self.ultimate_label_test.loc[star, 'period'] == "\\\\nodata"):
                index = self.ultimate_label_test.loc[star]
                self.ultimate_label_test.drop(star, axis=0, inplace=True)
            else:
                features1 = fats_fs.calculateFeature(valores.T).result().tolist()
                features2 = fats_fs.calculateFeature(valores2.T).result().tolist()
                features1.append(float(self.ultimate_label_test.loc[star, 'period']))
                self.ultimate_data_test_band1.append(features1.copy())
                features1.extend(features2)
                features_band2 = features2.copy()
                features_band2.append(float(self.ultimate_label_test.loc[star, 'period']))
                self.ultimate_data_test_band2.append(features_band2.copy())

    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


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
class classifier():
    def __init__(self):
        self.rf = None
        self.data = None
    def pickParameters(self, selection):
        aux = []
        for feats in self.data.ultimate_data_train:
            feats_aux = []
            for index in selection:
                feats_aux.append(feats[index])
            aux.append(feats_aux)
        self.data.ultimate_data_train = aux
        test_aux = []
        for feats in self.data.ultimate_data_test:
            feats_aux = []
            for index in selection:
                feats_aux.append(feats[index])
            test_aux.append(feats_aux)
        self.data.ultimate_data_test = test_aux
    def classifier(self, data, change_parameters = None, means = None):
        self.data = data
        if(change_parameters!=None):
            self.pickParameters(change_parameters)
        self.rf = RandomForestClassifier(n_estimators = 50, n_jobs=-1, random_state = 10, class_weight="balanced")
        values_train_final = data.ultimate_label_train['classALeRCE'].values
        onehot_values_train = []
        for j in values_train_final:
            cyka = [0,0,0,0,0,0]
            cyka[np.where(data.values == j)[0][0]] = 1
            onehot_values_train.append(cyka)
        p=np.random.permutation(len(onehot_values_train))
        if(means == None):
            try:
                self.data.ultimate_data_train = np.array(self.data.ultimate_data_train)[p]
            except:
                print("data not available")
        else:
            self.data.ultimate_data_train_band1 = np.array(self.data.ultimate_data_train_band1)[p]
            self.data.ultimate_data_train_band2 = np.array(self.data.ultimate_data_train_band2)[p]
            self.rf2 = RandomForestClassifier(n_estimators = 50, n_jobs=-1, random_state = 10, class_weight="balanced")
        onehot_values_train = np.array(onehot_values_train)[p]
        values_test_final = data.ultimate_label_test['classALeRCE'].values
        onehot_values_test = []
        for j in values_test_final:
            cyka = [0,0,0,0,0,0]
            cyka[np.where(data.values == j)[0][0]] = 1
            onehot_values_test.append(cyka)
        if means==None:
            self.rf.fit(self.data.ultimate_data_train, onehot_values_train)
            predictions = self.rf.predict(self.data.ultimate_data_test)
            onehot_values_test = np.array(onehot_values_test).argmax(1)
            predictions = np.array(predictions).argmax(1)
            plot_confusion_matrix(onehot_values_test, predictions, data.values)
        else:
            self.rf.fit(self.data.ultimate_data_train_band1, onehot_values_train)
            self.rf2.fit(self.data.ultimate_data_train_band2, onehot_values_train)
            predictions_one = self.rf.predict(self.data.ultimate_data_test_band1)
            predictions_two = self.rf2.predict(self.data.ultimate_data_test_band2)
            onehot_values_test = np.array(onehot_values_test).argmax(1)
            predictions = []
            for i in range(len(predictions_one)):
                aux = []
                for j in range(len(predictions_one[i])):
                    aux.append((predictions_one[i][j]+predictions_two[i][j])/2)
                predictions.append(aux)
            predictions = np.array(predictions).argmax(1)
            plot_confusion_matrix(onehot_values_test, predictions, data.values)
    def exportTree(self, features):
        estimator = self.rf.estimators_[0]
        export_graphviz(estimator, out_file='tree.dot',
                        feature_names=features,
                        class_names=self.data.values,
                        rounded=True, proportion=False,
                        precision=2, filled=True)
    def importanceVariable(self, features):
        feature_importances = self.rf.feature_importances_
        importance_order = np.argsort(-feature_importances)
        amount = 0
        for index in importance_order:
            amount += feature_importances[index]
            print('%s & %.3f' % (features[index], feature_importances[index]))
            if(amount>0.8):
                break

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
def generateDataFATS(filename="fats_processed.pkl", toFile=True, means=False):
    initialData = DataProcessor()
    initialData.generateLabelsPartition()
    if means:
        initialData.applyFATSTwoLists(feature_list, exclude_list)
    else:
         initialData.applyFATS(feature_list, exclude_list)
    if(toFile):
        initialData.save_object(filename)
# Función para abrir información en formato pkl. En el caso de que no exista, se genera y se abre otro.
def openDataFATSGenerado(filename, means=False):
    try:
        input = open(filename, 'rb')
        data = pickle.load(input)
    except:
        generateDataFATS(filename, means=means)
        data = openDataFATSGenerado(filename)
    return data
#La información puede ser obtenida mediante el uso de generateDataFATS con flag False en toFile, que lo corre cada vez que se ejecuta (No ocupa memoria), o
#generando un archivo para poder debuggear de inmediato la información.
toFile = True
means = True
if toFile:
    data = openDataFATSGenerado("fats_processed.pkl", means=means)
else:
    data = generateDataFATS(toFile=False, means=means)
## Por algún motivo existe un NaN a la hora de generar resultados, por lo cual se busco por aca. Es de suponer que
## Dado que ocurrió una vez puede ocurrir mas veces, esto es provisorio y por ende se generara un detector de nan
## Dsps
nansuspect = []
for a in data.ultimate_data_test_band2:
    for j in a:
        if(math.isnan(j)):
            nansuspect.extend(a)
index = data.ultimate_data_test_band2.index(nansuspect)
data.ultimate_data_test_band2.pop(index)
data.ultimate_data_test_band1.pop(index)
stars_training = data.ultimate_label_test.index.unique()
starnan = stars_training[index]
data.ultimate_label_test.drop(starnan, axis=0, inplace=True)
firstClassifier = classifier()
firstClassifier.classifier(data, means=means)
feature_list.append('Period')
feature_list2 = [
    'Amplitude2', 'AndersonDarling2', 'Autocor_length2',
    'Beyond1Std2', 'CAR_sigma2', 'CAR_mean2', 'CAR_tau2',
    'Con2', 'Eta_e2', 'Gskew2', 'MaxSlope2', 'Mean2',
    'Meanvariance2', 'MedianAbsDev2', 'MedianBRP2',
    'PairSlopeTrend2', 'PercentAmplitude2', 'Q312',
    'Rcs2', 'Skew2', 'SmallKurtosis2',
    'Std2', 'StetsonK2','Color']
#Para sacar el .dot del árbol y la importancia de las variables, eliminar los #
#firstClassifier.exportTree(feature_list)
firstClassifier.importanceVariable(feature_list)
plt.show()