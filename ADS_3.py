"""
ADS Assignment3 - Clustering and Fitting
created by : Chiamaka Ezenezi
"""

#importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import seaborn as sns
import sklearn.preprocessing as pp
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Defining relevant functions
def read_data(csv_filename):
    """reads a csv file into a dataframe
    and returns the original dataframe df1 and the
    cleaned transposed dataframe, df_T
    """
    
    #read original csv file into pandas dataframe
    df1 = pd.read_csv(csv_filename)

    #setting multi-index
    df1 = df1.set_index(["Series Name","Country Name"])
    
    #cleaning the data, drop irrelevant columns
    df1 = df1.drop(["Series Code", "Country Code"], axis=1)
    #drop last rows with Nan values
    df1 = df1.dropna()
    #get rid of '..' from the data
    df1 = df1.replace('..', np.nan)
    #assigning appropriate data type tothe dataframe
    df1 = df1.astype("float")
    #rename columns
    df1.columns = ["1970", "1971", "1972", "1973", "1974", "1975", "1976",
                   "1977", "1978", "1979", "1980", "1981", "1982", "1983",
                   "1984", "1985", "1986", "1987", "1988", "1989", "1990",
                   "1991", "1992", "1993", "1994", "1995", "1996", "1997", 
                   "1998", "1999", "2000"]
    #converting columns to numeric
    cols = df1.columns[2:]
    df1[cols] = df1[cols].apply(pd.to_numeric)

    #transposes the dataframe
    df_T = df1.transpose()
    #rename long column names
    df_T.rename(columns = 
    {("Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)"):
     "labor force participation rate", 
     ("Agriculture, forestry, and fishing, value added (annual % growth)"):
        "Agriculture value(annual % growth)", 
        ("Unemployment, total (% of total labor force) (modeled ILO estimate)"):
         "Unemployment(% of labor force"}, inplace=True)
    
    return (df1, df_T)


def plot_correlation_heatmap(year_data, title):
    """
    Plots a heatmap of the correlation among economic indicators for a given
    country.

    Parameters:
    - country_data: Pandas DataFrame containing economic indicators. 
      Rows represent different years, and columns represent different
      indicators.
    - title: Title of the heatmap.
    """
    # Calculate the correlation matrix
    correlation_matrix = year_data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(5, 3))

    # Create a heatmap using seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})

    # Set title and labels
    plt.title(title)
    plt.xlabel('Economic Indicators')
    plt.ylabel('Economic Indicators')
    # Display the plot
    plt.show()

     
def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object

    kmeans.fit(xy) # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    
    return score



def plot_kmeans_clusters(df, n_clusters=3):
    """
    Plot a scatter plot with KMeans clustering.

    Parameters:
    - df_to_cluster: DataFrame containing data to be clustered.
    - n_clusters: Number of clusters for KMeans.
    """
    # Extract x and y values of data points
    x = df["Manufacturing, value added (annual % growth)"]
    y = df["GDP growth (annual %)"]

    # Normalize the data
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(df)

    # Set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)
    
    # Extract cluster labels
    labels = kmeans.labels_
    
    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))
    
    # Plot data with kmeans cluster number
    plt.scatter(x, y, 10, labels, marker="o", cmap='Paired')
    
    # adding the cluster memebership information to the dataframe
    df_to_cluster["labels"] = labels
    
    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
    plt.xlabel("% growth of value added by manufacturing")
    plt.ylabel("GDP growth (annual %)")
    plt.show()


#calling the read_data functions
csv_filename = "world_indicators_Data.csv"
original_data, transposed_data = read_data(csv_filename)
print("Original Data:")
original_data
print("\nTransposed Data:")
transposed_data

one_indicator = transposed_data[['GDP per capita (constant LCU)']].copy()

#this helped to show the year with fewer Nan values, by changing the years
one_year_data = pd.pivot_table(original_data, values="1995",
                               index= "Country Name", columns= "Series Name")
one_year_data.isna().sum()

plot_correlation_heatmap(one_year_data, title="1995")

one_year_data = one_year_data.dropna()

#create a scalar object
scaler = pp.RobustScaler()

#extract the two columns
df_to_cluster = one_year_data[["Manufacturing, value added (annual % growth)",
                                "GDP growth (annual %)"]].copy()
#let's examine the mini dataframe
scaler.fit(df_to_cluster)

df_norm = scaler.transform(df_to_cluster)

plt.figure()

plt.scatter(df_norm[:,0], df_norm[:,1], 10, marker='o')

plt.xlabel('% growth of value added by Manufacturing')
plt.ylabel('GDP growth (annual %)')

plt.show()

# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

#the silhouette score suggests 2 clusters but lets take 3 instead
plot_kmeans_clusters(df_to_cluster, n_clusters=3)

#Now we have 3 clusters of countries:
#high for countries with high GDP growth contributed by high manufaturing,
#middle and low producing countries

print(df_to_cluster)