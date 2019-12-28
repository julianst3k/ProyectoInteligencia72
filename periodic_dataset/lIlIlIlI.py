import pandas as pd
import matplotlib.pyplot as plt


detections = pd.read_pickle('periodic_detections.pkl')
labels = pd.read_pickle('periodic_labels.pkl')

print(detections.head())
print(labels.head())

# Plot an object

for stars in labels.index.unique():
    first_lc = detections.loc[stars]
    first_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
    flc1 = first_lc[first_lc.fid == 1].sort_values('mjd').drop_duplicates('mjd')
    second_lc = first_lc[(first_lc.sigmapsf_corr < 1) & (first_lc.sigmapsf_corr > 0)]
    flc2 = second_lc[first_lc.fid == 2].sort_values('mjd').drop_duplicates('mjd')
    flc1.dropna()
    flc2.dropna()
    valores = flc1[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
    valores2 = flc2[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values
    print(len(valores), len(valores2))
    if (len(valores) < 8 and len(valores2) > 15):
        first_object_oid = stars
        break

first_object_detections = detections.loc[first_object_oid]

print(f'Object {first_object_oid} has {len(first_object_detections)} detections')

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(
    first_object_detections.mjd,
    first_object_detections.magpsf_corr,
    c=first_object_detections.fid)

# In astronomy we plot the magnitude axis inverted (higher magnitude, dimmer objects)
plt.gca().invert_yaxis()
plt.xlabel('Time [mjd]')
plt.ylabel('Magnitude')
plt.title(f'{first_object_oid} light curve')

plt.subplot(1, 2, 2)
period = float(labels.loc[first_object_oid].period)
plt.scatter(
    (first_object_detections.mjd.values % period)/period,
    first_object_detections.magpsf_corr,
    c=first_object_detections.fid)

# In astronomy we plot the magnitude axis inverted (higher magnitude, dimmer objects)
plt.gca().invert_yaxis()
plt.xlabel('Phase')
plt.ylabel('Magnitude')
plt.title(f'{first_object_oid} light curve (folded with period {period:.3f} days)')
plt.show()