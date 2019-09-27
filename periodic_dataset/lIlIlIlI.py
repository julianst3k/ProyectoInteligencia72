import pandas as pd
import matplotlib.pyplot as plt


detections = pd.read_pickle('periodic_detections.pkl')
labels = pd.read_pickle('periodic_labels.pkl')

print(detections.head())
print(labels.head())

# Plot an object
first_object_oid = detections.index.values[0]
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