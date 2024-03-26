from scipy import fftpack  # or 'fft'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

df_daily_activity = pd.read_csv("daily_cp_activity_dataset.csv")
df_daily_activity['date'] = pd.to_datetime(df_daily_activity['date'])

anomaly_threshold = 500
df_daily_activity_cleaned = df_daily_activity.copy(deep= True)
df_daily_activity_cleaned.loc [df_daily_activity['courier_partners_online']>500, 'courier_partners_online'] = None
table = df_daily_activity_cleaned.ffill()

plt.figure(figsize=(10, 2))  
plt.plot (table["date"], table["courier_partners_online"], 'k--',marker='o', label = "truth")
plt.ylabel("Courier number")

today = '2022-11-20'  # "Today"'s date for the prediction
today = '2021-11-20'  # "Today"'s date for the prediction

l_past = 14  # base approximation on Lpast days
l_future = 14  # use same approximation of following Lfuture days
l = l_past + l_future  # total length of approximation

table_today = table[table['date'] < today]  # use only data up to "today"

dates = table_today["date"]
couriers = table_today["courier_partners_online"]

# get relative date from the first day
#t_full = np.array((dates - dates.iloc[0]).dt.days)

first = np.maximum(0, table_today.shape[0] - 201)  # first day of each l-day window from 100 days "ago" to "today"
end = table_today.shape[0] - l - l_future + 1
indices = [i for i in range(first, end)]

# initialize block concatenation array for an individual country
all_blocks = np.zeros((l, len(indices)))
counter = 0

for s1 in indices:  # loop over all selected windows of L days

    z = couriers[s1:s1 + l]  # data within a window
    
    all_blocks[:, counter] = z #np.log(ped + z)
    counter += 1
 
all_blocks -= np.mean(all_blocks, axis=1, keepdims=True)  # subtract sample mean
u, s, v = np.linalg.svd(all_blocks @ all_blocks.T)  # SVD of covariance matrix (ignoring 1/(size(allBlocks,2)-1)*)

# number of components
j=10

m = u[:, 0:j]  # build frame matrix from first J principal vectors
#t = np.array((dates - dates.iloc[0]).dt.days)

t = pd.date_range(start=dates.iloc[-1] - pd.Timedelta(days=l_past-1), 
                     end=dates.iloc[-1] + pd.Timedelta(days=l_future ))
# weights are zero on the last Lfuture days
w = np.concatenate([[0.5], np.ones((l_past-2,)), [0.5], np.zeros((l_future,))], axis=0)
couriers_last = np.concatenate( [couriers[-(l_past):],  np.zeros((l_future,))], axis=0)
      
# the dual frame  (computed with respect to weighted inner product)
m_tilde = m @ np.linalg.pinv(m.T * np.atleast_2d(np.ravel(w)) @ m)

hatz = (m @ m_tilde.T @ (w *  couriers_last)) 

#plt.plot(t, hatz, 'r.-', linewidth=3)  # plot approximation of the future

plt.plot(t[0:l_past ], hatz[0:l_past ], 'b.-', linewidth=3, marker='o', label = "fit")  # Blue markers for the approximation of the past
plt.plot(t[l_past:], hatz[l_past:], 'r.-', linewidth=3, marker='o', label = "prediction")  # Red markers for the approximation of the future

plt.xlim(t[0],t[-1])

plt.legend()

        # plot boundary line past/future
#plt.plot([t[l_past], t[l_past]],
#                [np.minimum(np.min(hatz), np.min(z)), np.maximum(np.max(hatz), np.max(z))], 'k--')

plt.show()