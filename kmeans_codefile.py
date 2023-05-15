'''Analyzing Top 20 Countries' Greenhouse Gas Emissions: K-Means Clustering 
and Exponential Growth Models for India and China'''
'''Student Name: Muhammad Junaid Saif'''
'''Student ID: 22030494'''
''' GITHUB Link: https://github.com/ms22akc/Assignment3_poster '''

#-----------------------------------------------------#
#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns


# load the data
df = pd.read_csv(r'C:\Users\M_jun\Downloads\poster\Assignment3_poster\totalgreenhousegas1.csv')
df.head()
df.describe()

#------------------------------------------------
#CORRELATION HEATMAP
# extract the desired years for correlation heatmap

years = ['1990', '1996', '2002', '2008', '2014', '2019']
df_cor = df[years]

# calculate the correlation matrix
corr_matrix = df_cor.corr()

# create the heatmap
sns.set(font_scale=1.2)
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Year', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#--------------------------------------------------
#TIME SERIES PLOT 

# Set the Country column as the index
data = df.set_index('Country Name')

# Define a color palette
palette = sns.color_palette('muted', n_colors=len(data))

# Plot all countries' time series on the same plot
plt.figure(figsize=(12, 6), dpi=300)
for i, country in enumerate(data.index):
    plt.plot(data.columns, data.loc[country], color=palette[i], label=country, linewidth=1.5)

# Customize the plot
plt.legend(fontsize=10, framealpha=0.7, ncol=3)
plt.title('Greenhouse Gas Emissions by Country', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Emissions (kt of CO2 equivalent)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, rotation=45, ha='right', fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

# Show the plot
plt.show()

#------------------------------------------------
#K-MEANS CLUSTERING 

# Extract columns of interest
df_cluster = df[['Country Name', '1999', '2019']].copy()

# Normalization
scaler = StandardScaler()
df_cluster_norm = scaler.fit_transform(df_cluster[['1999', '2019']])

# Compute silhouette score
# Set up empty lists for the scores and the number of clusters
scores = []
clusters = []

# Iterate over different numbers of clusters
for n_clusters in range(2, 11):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df_cluster_norm)
    
    # Calculate the silhouette score
    score = silhouette_score(df_cluster_norm, labels)
    
    # Append the score and number of clusters to the lists
    scores.append(score)
    clusters.append(n_clusters)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(clusters, scores, marker='o')
ax.set_xlabel('Number of clusters', fontsize=14)
ax.set_ylabel('Silhouette score', fontsize=14)
ax.set_title('Silhouette score vs. Number of Clusters', fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.grid(alpha=0.3)

plt.show()

# perform clustering
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(df_cluster_norm)

# set up color palette
palette = sns.color_palette('bright', n_colors=len(np.unique(clusters)))

# add color column to the dataframe
df_cluster['Color'] = [palette[i] for i in clusters]

# plot clusters with annotations
ax = df_cluster.plot(kind='scatter', x='1999', y='2019', c=df_cluster['Color'], s=60, alpha=0.8, figsize=(12, 8))
ax.set_xlabel('GHG Emissions in 1999', fontsize=14, fontweight='bold')
ax.set_ylabel('GHG Emissions in 2019', fontsize=14, fontweight='bold')
ax.set_title('GHG Emissions by Country: K-Means Clustering', fontsize=16, fontweight='bold')

# add annotations
for i, txt in enumerate(df_cluster['Country Name']):
    ax.annotate(txt, (df_cluster['1999'][i], df_cluster['2019'][i]), fontsize=10)

# customize legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = np.unique(clusters)
legend_handles = [handles[np.where(clusters == label)[0][0]] for label in unique_labels]
legend_labels = ['Cluster {}'.format(label) for label in unique_labels]
ax.legend(legend_handles, legend_labels, fontsize=12, loc='upper left', framealpha=0.8)

# show plot
plt.show()

#------------------------ (b) -------------------------------
# SIMPLE MODEL FITTING 
import scipy.optimize as opt

data = df
countries = ['CHN', 'IND']
country_data = data[data['Country Name'].isin(countries)].set_index('Country Name').T

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""     
    f = n0 * np.exp(g*(t-1990))
    return f

'''FOR CHINA'''
# Subset the data for CHN
chn_data = country_data['CHN'].reset_index()
chn_data.columns = ['Year', 'GHG']
chn_data['Year'] = pd.to_numeric(chn_data['Year']) # convert 'Year' column to numeric type

# create x values (years)
x = chn_data['Year']

param, covar = opt.curve_fit(exponential, x, chn_data['GHG'], p0=(1.2e12, 0.03))
print("GHG 1990", param[0]/1e9)
print("growth rate", param[1])

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# Plot the data and exponential fit
plt.plot(x, chn_data['GHG'], 'o', label="Actual data")
plt.plot(x, exponential(x, *param), '--', label="Exponential fit")
# Add labels and legend
plt.xlabel("Year", fontsize=14, fontweight='bold')
plt.ylabel("GHG Emissions (kt of CO2 equivalent)", fontsize=14, fontweight='bold')
plt.title("GHG Emissions in China from 1990-2019", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
# Customize tick labels
plt.xticks(x, rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
# Add grid
plt.grid(linestyle='--')
# Show plot
plt.show()

#Forecasting
year = np.arange(1960, 2031)
sigma = np.sqrt(np.diag(covar))
forecast = exponential(year, *param)
plt.figure()
plt.plot(chn_data["Year"], chn_data["GHG"], label="GHG Emission")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()


import errors as err
low, up = err.err_ranges(year, exponential, param, sigma)
plt.figure()
plt.plot(chn_data["Year"], chn_data["GHG"], label="GHG Emission")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()


''' FOR USA'''
# Subset the data for USA
ind_data = country_data['IND'].reset_index()
ind_data.columns = ['Year', 'GHG']
ind_data['Year'] = pd.to_numeric(ind_data['Year']) # convert 'Year' column to numeric type

# create x values (years)
x = ind_data['Year']

# plot the data
plt.plot(x, ind_data['GHG'])
plt.title('GHG Emissions in India (1990-2019)')
plt.xlabel('Year')
plt.ylabel('GHG Emissions (kt of CO2 equivalent)')
plt.show()


param, covar = opt.curve_fit(exponential, x, ind_data['GHG'], p0=(1.2e12, 0.03))
print("GHG 1990", param[0]/1e9)
print("growth rate", param[1])

# set plot style
sns.set_style('whitegrid')
# create figure and axes objects
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# plot the exponential fit
ax.plot(x, exponential(x, *param), color='tab:orange', label="Exponential fit", linewidth=2)
# plot the actual data
ax.plot(x, ind_data['GHG'], color='tab:blue', label="Actual data", linewidth=2)
# set axis labels and title
ax.set_xlabel("Year", fontsize=12, fontweight='bold')
ax.set_ylabel("GHG Emissions (kt of CO2 equivalent)", fontsize=12, fontweight='bold')
ax.set_title('GHG Emissions in India (1990-2019)', fontsize=14, fontweight='bold')
# customize tick labels and legend
ax.tick_params(axis='both', labelsize=10)
ax.legend(fontsize=10)
# display plot
plt.show()

# calculate residuals between predicted and actual values
residuals = ind_data['GHG'] - exponential(x, *param)

# calculate standard deviation of residuals
residual_std = np.std(residuals)

# generate predictions for the next 10 years
x_pred = np.arange(2020, 2030)
y_pred, pred_err = opt.curve_fit(exponential, x, ind_data['GHG'], p0=(1.2e12, 0.03), sigma=residual_std, absolute_sigma=True)(x_pred)

# print predicted values and confidence intervals
print('Predicted GHG Emissions (kt of CO2 equivalent) for 2020-2029:')
for i in range(len(x_pred)):
    print(f'{x_pred[i]}: {y_pred[i]} ± {pred_err[i]}')


#Forecasting
year = np.arange(1960, 2031)
sigma = np.sqrt(np.diag(covar))
forecast = exponential(year, *param)
plt.figure()
plt.plot(ind_data["Year"], ind_data["GHG"], label="GHG Emission")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()


import errors as err
low, up = err.err_ranges(year, exponential, param, sigma)
plt.figure()
plt.plot(ind_data["Year"], ind_data["GHG"], label="GHG Emission")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()


#--------------------------------------------
#BAR CHART 
df2 = df

# dictionary mapping country codes to full names
country_names = {
    'AUS': 'Australia', 'BRA': 'Brazil', 'CHN': 'China', 'IND': 'India', 'IDN': 'Indonesia', 'JPN': 'Japan',
    'MEX': 'Mexico', 'RUS': 'Russian Federation', 'USA': 'United States', 'CAN': 'Canada', 'EUU': 'European Union',
    'IRN': 'Iran, Islamic Rep.', 'DEU': 'Germany', 'FRA': 'France', 'ITA': 'Italy', 'SAU': 'Saudi Arabia',
    'AFE': 'Africa Eastern and Southern', 'SSD': 'South Sudan', 'SAS': 'South Asia', 'KOR': 'Korea, Rep.'
}

# replace country codes with full names
df2['Country Name'] = df2['Country Name'].apply(lambda x: country_names[x])

# select only numerical columns
num_cols = [col for col in df2.columns if col != 'Country Name']
df2['Average'] = df2[num_cols].mean(axis=1)

# create bar chart
sns.set(style='whitegrid')
plt.figure(figsize=(12, 8), dpi=300)
ax = sns.barplot(x='Average', y='Country Name', data=df2)
ax.set(xlabel='Average GHS emitted', ylabel=None)
ax.tick_params(axis='both', which='major', labelsize=12, width=2, pad=5, length=5)
ax.xaxis.label.set_fontweight('bold')
ax.yaxis.label.set_fontweight('bold')
plt.yticks(rotation=45, ha='right')
plt.show()