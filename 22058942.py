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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import importlib as imlib
import errors as err
import scipy.optimize as opt

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
    {"Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)":
     "labor force participation rate", 
     "Agriculture, forestry, and fishing, value added (annual % growth)":
        "Agriculture value(annual % growth)", 
        "Unemployment, total (% of total labor force) (modeled ILO estimate)":
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

def plot_kmeans_clusters(df, n_clusters=3, title=None):
    """
    Plot a scatter plot with KMeans clustering.

    Parameters:
    - df_to_cluster: DataFrame containing data to be clustered.
    - n_clusters: Number of clusters for KMeans.
    - title: title of the plot
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
    plt.title(title)
    plt.legend(  )
    plt.show()


def plot_multiple_lines(dataframes, y, title, legend_labels, colors):
    """
    Plot multiple lines on the same graph using Matplotlib.

    Parameters:
    - dataframes: List of DataFrames. Each DataFrame should have columns 'x' and 'y'.
    - y : column for the y-axis.
    - title: Title for the plot.
    - legend_labels: Optional list of legend labels for each line.
    - color : color of the line plot
    """
    plt.figure(figsize=(10, 6))

    for i, df in enumerate(dataframes):
        x_values = df.index
        x_values = x_values.astype('int')
        y_values = df[y]
        
        color = colors[i] if colors else None
        label = legend_labels[i] if legend_labels else f'Line {i + 1}'
        plt.plot(x_values, y_values, marker='o', linestyle='-', color=color, label=label)
    
     # Set x-axis ticks for every 5 years
    plt.xticks(range(min(x_values), max(x_values)+1, 5))
    plt.xlabel("years")
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
 

def linfunc(x, a, b):
    """ Function for fitting
        x: independent variable
        a, b: parameters to be fitted
    """
    
    y = a*x + b
    
    return y
    

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f


def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
        
    # makes it easier to get a guess for initial parameters
    t = t - 1985
        
    f = n0 * np.exp(g*t)
        
    return f

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

plt.xlabel('Manufacturing, value added (annual % growth)')
plt.ylabel('GDP (annual % growth)')

plt.show()

# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

#the silhouette score suggests 2 clusters but lets take 3 instead
plot_kmeans_clusters(df_to_cluster, n_clusters=3, 
  title='country cluster based on GDP % growth and manufacturing contribution')

#Now we have 3 clusters of countries:
#high for countries with high GDP growth contributed by high manufaturing,
#middle and low producing countries

print(df_to_cluster)

high_producing_countries = df_to_cluster[df_to_cluster['labels']==2]
medium_producing_countries = df_to_cluster[df_to_cluster['labels']==1]
low_producing_countries = df_to_cluster[df_to_cluster['labels']== 0]

#picking country from each of the cluster we check the historic GDP growth
Ukraine_data = transposed_data.xs('Ukraine', level="Country Name", axis=1)
Nigeria_data = transposed_data.xs('Nigeria', level="Country Name", axis=1)
Bangladesh_data = transposed_data.xs('Bangladesh', level="Country Name", axis=1)
Sweden_data = transposed_data.xs('Sweden', level="Country Name", axis=1)
Netherlands_data = transposed_data.xs('Netherlands', level="Country Name", axis=1)
Japan_data = transposed_data.xs('Japan', level="Country Name", axis=1)

dataframes = [Nigeria_data, Bangladesh_data, Netherlands_data]
legend_labels = ['low', 'medium', 'high']
color = ['red', 'orange', 'blue']
plot_multiple_lines(dataframes, 'GDP growth (annual %)', 'GDP historical growth',
                    legend_labels, color)

dataframes = [Ukraine_data, Sweden_data, Japan_data]
legend_labels = ['low', 'medium', 'high']
color = ['red', 'orange', 'blue']
plot_multiple_lines(dataframes, 'GDP growth (annual %)', 'GDP historical growth',
                    legend_labels, color)

#one of the high_producing countries is selected to fit
#sub dataframe 
Bangladesh_data = Bangladesh_data.reset_index().rename(columns={'index': 'years'})
#print(Bangladesh_data)

Bangladesh_gdp_growth = Bangladesh_data[['years','GDP growth (annual %)']]
Bangladesh_gdp_growth['years'] = Bangladesh_gdp_growth['years'].astype(int)
print(Bangladesh_gdp_growth.dtypes)

#let's inspect the dataset
Bangladesh_gdp_growth.plot("years", "GDP growth (annual %)")
plt.show()

xdata = Bangladesh_gdp_growth['years']
ydata = Bangladesh_gdp_growth['GDP growth (annual %)']
#inspect with exponential function
param, pcovar = opt.curve_fit(exponential, xdata, ydata)

#extract the columns to model and fit
xdata = Bangladesh_gdp_growth['years']
ydata = Bangladesh_gdp_growth['GDP growth (annual %)']
param, pcovar = opt.curve_fit(exponential, xdata, ydata)

# list of parameters
print("parameters:", param)
print("covariance-matrix")
print(pcovar)

sigma = np.sqrt(np.diag(pcovar))

print(f"a = {param[0]:5.3f} +/- {sigma[0]:5.3f}")
print(f"b = {param[1]:5.3f} +/- {sigma[1]:5.3f}")

#plot the fit trial and keep adjusting the parameters to get the best fit
plt.figure()
plt.plot(xdata, exponential(xdata,3.185 , 0.035), label ="fit")
plt.plot(xdata, ydata)

plt.show()

#fitted data
Bangladesh_gdp_growth["fit"] = exponential(xdata, 3.185 , 0.035 )
Bangladesh_gdp_growth.plot('years', ["GDP growth (annual %)", "fit"])

plt.show()

#plot fit trial with logistics function
param1, covar = opt.curve_fit(logistic, xdata, ydata)

# list of parameters
print("parameters:", param1)
print("covariance-matrix")
print(pcovar)

var = np.diag(covar)
sigma = np.sqrt(var)
print(f"turning point {param1[2]: 6.1f} +/- {sigma[2]: 4.1f}")
print(f"GDP at turning point {param1[0]: 7.3e} +/- {sigma[0]: 7.3e}")
print(f"growth rate {param1[1]: 6.4f} +/- {sigma[1]: 6.4f}")

plt.figure()
plt.plot(xdata, logistic(xdata,3.324 , 1, 1), label ="fit")
plt.plot(xdata, ydata)

plt.show()

#the exponential function is better fit,so will be using it for our prediction
#lets now predict for 30 mre years, up to 2030

imlib.reload(err)

year = np.linspace(1960, 2030, 100)
forecast = exponential(year, 3.185 , 0.035)
sigma = err.error_prop(year, exponential, [3.185 , 0.035], covar)
up = forecast + sigma
low = forecast - sigma
plt.figure()
Bangladesh_gdp_growth.plot('years', "GDP growth (annual %)", label="GDP growth")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP growth")
plt.legend()
plt.show()
