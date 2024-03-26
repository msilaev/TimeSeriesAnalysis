import numpy as np  # Numpy for arrays
import matplotlib.pyplot as plt  # PyPlot for figures
import pandas as pd  # Pandas is a library for handling all kinds of data tables
from scipy import fftpack  # or 'fft'

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data-old.csv"
table = pd.read_csv(url, sep=",")  # fetch data from web
today = '2021-11-20'  # "Today"'s date for the prediction
l_past = 14  # base approximation on Lpast days
l_future = 14  # use same approximation of following Lfuture days
l = l_past + l_future  # total length of approximation

countries = table["location"].unique()  # ALL countries to be used for computing PCA
table["date"] = pd.to_datetime(table["date"]) # Change date column to datetime

table_today = table[table['date'] < today]  # use only data up to "today"
all_blocks = []  # we collect blocks of length L days from ALL listed countries

for jj in range(len(countries)):  # outer loop over ALL listed countries
    print(countries[jj], end=', ')
    idx = table_today["location"] == countries[jj]  # indices for data for this country name
    z_full = table_today[idx]["new_cases"]  # extract new-cases data from table
    dates = table_today[idx]["date"]
    if dates.empty: # No data
        continue

    # get relative date from the first day
    t_full = np.array((dates - dates.iloc[0]).dt.days)
    t_full = t_full[~z_full.isnull()]  # REMOVE NaNs (usually first few and last few days in the series)
    z_full = z_full[~z_full.isnull()]

    # consider only if enough data to put at least one window of L days, ignoring last Lfuture days.
    if t_full.size < l + l_future:
        continue

    first = np.maximum(0, z_full.shape[0] - 101)  # first day of each l-day window from 100 days "ago" to "today"
    end = z_full.shape[0] - l - l_future + 1
    indices = [i for i in range(first, end)]

    # initialize block concatenation array for an individual country
    all_blocks_country = np.zeros((l, len(indices)))
    counter = 0

    for s1 in indices:  # loop over all selected windows of L days
        z = z_full[s1:s1 + l]  # data within a window

        # pedestal for log transform (1% of max, unless min is negative)
        ped = np.max(z) / 100 - np.minimum(0, np.min(z))
        # will skip the window if z identically zero, new infection data is processed in log scale
        if ped > 0:
            # increase counter and collect window content
            all_blocks_country[:, counter] = np.log(ped + z)
            counter += 1

    all_blocks.append(all_blocks_country[:, :counter])  # concatenate new completed country

# Change all_blocks from l x count x num_countries list to l x (count * num_countries) array
all_blocks = np.concatenate(all_blocks, axis=1)
all_blocks -= np.mean(all_blocks, axis=1, keepdims=True)  # subtract sample mean
u, s, v = np.linalg.svd(all_blocks @ all_blocks.T)  # SVD of covariance matrix (ignoring 1/(size(allBlocks,2)-1)*)

subfig_h = int(np.round(np.sqrt(l * 3)))
subfig_v = int(np.ceil((l + np.ceil(subfig_h / 3)) / subfig_h))

plt.figure(figsize=(18,8))
plt.subplot(subfig_v, 3, subfig_v * 3)  # subfigH*subfigV subplot grid
plt.plot(s, 'r.-')
plt.yscale('log')
plt.title('singular values')
for jj in range(0, l):  # subplots for singular vectors
    plt.subplot(subfig_v, subfig_h, jj + 1)
    plt.plot(u[:, jj], 'b.-')
    plt.title(jj)
plt.show()

from datetime import timedelta

countries_fig = ['World', 'Finland', 'Italy', 'Germany', 'Australia', 'Brazil', 'Bangladesh', 'Colombia',
                 'South Korea', 'United States', 'United Kingdom', 'India', 'China', 'Iran', 'Russia']

table_extr = table[(table['date'] <= pd.to_datetime(today)+timedelta(days=l_future)) & 
                   (table['date'] > pd.to_datetime(today)-timedelta(days=l_past))]


plt.figure(figsize=(12,10))
for jc in range(len(countries_fig)):
    # indices for country by comparing location strings, ignore null
    idx = table_extr["location"] == countries_fig[jc]
    z = table_extr[idx]["new_cases"]  # extract new-cases data from table
    dates = pd.to_datetime(table_extr[idx]["date"])

    dates = dates[~z.isnull()]
    z = z[~z.isnull()]

    # Convert dates to strings for display
    dates_disp = dates.apply(lambda x: x.strftime('%d-%b-%Y'))

    # get relative date from the first day
    t = np.array((dates - dates.iloc[0]).dt.days)

    dates_l = dates_disp.iloc[-l:]  # recent dates
    plt.subplot(3, int(np.ceil(len(countries_fig) / 3)), jc + 1)
    plt.plot(t, z, 'b.-', linewidth=4)
    plt.title(countries_fig[jc])  # plot data
    plt.xticks([t[0], t[l_past - 1], t[l_past + l_future - 1]],
               [dates_l.iloc[0], dates_l.iloc[l_past - 1], dates_l.iloc[l_past + l_future - 1]], rotation=-25,
               fontsize=5)
    plt.yticks(fontsize=5)

    for j in [6]:  # frame and dual frame matrices to be build using only the first J components

        m = u[:, 0:j]  # build frame matrix from first J principal vectors

        if 0:  # orthonormal polynomials of order J-1 --- can be enabled to compare vs PCA
            # scaling of indep. variable for numerics
            t_scaled = (t - np.min(t)) / (np.max(t) - np.min(t)) * 2 - 1
            # orthonormalization here helps numerics when computing Mtilde below
            [m, r] = np.linalg.qr(np.atleast_2d(t_scaled).T ** np.array(np.arange(0, j), ndmin=2))
        elif 0:  # DCT --- can be enabled to compare vs PCA
            m = fftpack.idct(np.eye(l, j), axis=0, norm='ortho')

        # weights are zero on the last Lfuture days
        w = np.concatenate([[0.5], np.ones((l_past-2,)), [0.5], np.zeros((l_future,))], axis=0)
        # s = 2; w = np.exp(-((0.5/s ** 2) * (np.minimum(0, t - (t[0] + l_past - 14))) ** 2)) # Gaussian window
        # w[l_past:] = 0

        # the dual frame  (computed with respect to weighted inner product)
        m_tilde = m @ np.linalg.pinv(m.T * np.atleast_2d(np.ravel(w)) @ m)

        # pedestal for log transform (1% of max, unless min is negative)
        ped = np.max(z) / 100 - np.minimum(0, np.min(z))
        #  analysis and synthesis in logarithmic scale
        hatz = np.exp(m @ m_tilde.T @ (w * np.log(ped + z))) - ped

        plt.plot(t[0:l_past + 1], hatz[0:l_past + 1], 'r.-', linewidth=6)  # plot approximation of the past
        plt.plot(t[l_past:], hatz[l_past:], 'r.-', linewidth=3)  # plot approximation of the future
        # plot boundary line past/future
        plt.plot([t[l_past], t[l_past]],
                 [np.minimum(np.min(hatz), np.min(z)), np.maximum(np.max(hatz), np.max(z))], 'k--')

plt.show()