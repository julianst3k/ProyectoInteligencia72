import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
y1 = [26000, 11000, 3424]
y2 = [15000,6600,2848]
x1 = [2170454,944963,372367]
x = [[0.954312,0.963944],[0.955102,0.973806]]
df_cm = pd.DataFrame(x,index = [i for  i in ["3","10"]], columns = [i for i in ["7","50"]])
plt.figure(figsize = (10,7))
ax= plt.subplot()
ax.plot(x1,y1,'-o', label="Sin índices")
ax.plot(x1,y2,'-o', label="Con índices")
ax.set_xlabel("Tuplas")
ax.set_ylabel("Tiempo [ms]")
ax.set_title("Tiempo vs Tuplas")
plt.legend(bbox_to_anchor=(0.8, 0.95), loc='upper left',
            borderaxespad=0.)
plt.show()