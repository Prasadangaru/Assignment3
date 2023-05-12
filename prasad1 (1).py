#importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score


# Read the dataset
df = pd.read_csv('API_SP.RUR.TOTL_DS2_en_csv_v2_5363648.csv', header=2)

# Use Functions From lecture class

def scaler(df):   
   # Use the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max

def backscale(arr, df_min, df_max):
    
    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1970, 2010)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalizing the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True) # replace NaN values with 0

# Find the optimal number of clusters using the silhouette method
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(df_norm)
    silhouette_scores.append(silhouette_score(df_norm, cluster_labels))

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Using K-means clustering to group data
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)


#Checking the years in the list
print(df_years.columns)



# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['2009'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('Rural Population Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('Rural Population in 2009')

# Add legend
plt.legend()

# Show the plot
plt.show()


# Display countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(cluster_countries)
    print()

def linear_model(x, a, b):
    return a*x + b

# Define the columns to use
columns_to_use = [str(year) for year in range(1970, 2010)]


# Select a country
country = 'Europe & Central Asia'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values[0]
x_data = np.array(range(1960, 2000))
y_data = country_data

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)

#applying functiion from the lecture class
def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper

# Predicting future values and corresponding confidence intervals
x_future = np.array(range(1960, 2045))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, alpha=0.3, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel('Rural Population')
plt.title(f'{country} Rural Population Fitting')
plt.legend()
plt.grid(True)
plt.show()




# Load the data
df = pd.read_csv('API_SP.RUR.TOTL_DS2_en_csv_v2_5363648.csv', skiprows=4)

# Select the columns for the years between 2015-2021
years = [str(year) for year in range(2015, 2022)]
cols = ['Country Name', 'Country Code'] + years
df = df[cols]

# Filter the data for the selected countries
countries = ['China', 'India', 'Afghanistan', 'South Africa']
df = df[df['Country Name'].isin(countries)]

# Melt the DataFrame to long format
id_vars = ['Country Name', 'Country Code']
value_vars = years
df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Rural Population')

# Create a bar graph
fig, ax = plt.subplots(figsize=(12, 8))
df.groupby(['Country Name', 'Year'])['Rural Population'].sum().unstack().plot(kind='bar', ax=ax)

# Set the title and axis labels
plt.title('Rural Population for Selected Countries, 2015-2021')
plt.xlabel('Countries')
plt.ylabel('Rural Population')

# Add grid
ax.grid(axis='y', linestyle='--')

# Show the plot
plt.show()


# Box plot for cross-comparison between clusters
plt.figure(figsize=(12, 8))
df_years.boxplot(column='2009', by='Cluster', vert=False)
plt.title('Cross-comparison between Clusters')
plt.xlabel('Rural Population in 2009')
plt.ylabel('Cluster')
plt.show()


