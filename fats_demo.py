import pandas as pd
import FATS



detections = pd.read_pickle('periodic_dataset/periodic_detections.pkl')
print(detections.head())

oids = detections.index.unique()
first_lc = detections.loc[[oids[0]]]

print('Object', oids[0], 'has', len(first_lc), 'detections')

feature_list = [
    'Amplitude', 'AndersonDarling', 'Autocor_length',
    'Beyond1Std', 'CAR_sigma', 'CAR_mean', 'CAR_tau',
    'Con', 'Eta_e', 'Gskew', 'MaxSlope', 'Mean',
    'Meanvariance', 'MedianAbsDev', 'MedianBRP',
    'PairSlopeTrend', 'PercentAmplitude', 'Q31',
    'Rcs', 'Skew', 'SmallKurtosis',
    'Std', 'StetsonK']

fats_fs = FATS.FeatureSpace(
    Data=['magnitude', 'time', 'error'],
    featureList=feature_list,
    excludeList=[
        'StetsonJ',
        'StetsonL',
        'Eta_color',
        'Q31_color',
        'Color'])

first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]

flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
flc2 = first_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')

print(flc1.shape)
print(flc2.shape)

data = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
features1 = fats_fs.calculateFeature(data.T).result().tolist()

for name, value in zip(feature_list, features1):
    print(name, value)

data = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
features2 = fats_fs.calculateFeature(data.T).result().tolist()

for name, value in zip(feature_list, features2):
    print(name, value)
