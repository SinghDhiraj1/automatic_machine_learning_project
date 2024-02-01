import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.utils import resample

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier  

import warnings
warnings.filterwarnings('ignore')

import re
from copy import deepcopy

import pycountry
import geonamescache
from geonamescache import GeonamesCache
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_auc_score,f1_score,classification_report,precision_recall_curve 

from flaml import AutoML

import shutil
import os
import pickle
from pathlib import Path
import logging
logging.basicConfig(filename="AutoML.log",level=logging.DEBUG,
                    # '{%(pathname)s:%(lineno)d}' reference for adding pathname in logging
                    format= '[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

def info(df):
    
    # Display the number of rows, columns, and memory usage of the DataFrame.
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Memory Usage: ", df.memory_usage(deep=True).sum(), ' Bytes')
    print('\n\n')
    
    # Show the data types of each column.
    print(df.dtypes)
    print('\n\n')
    
    # Analyze missing values by counting them and calculating their percentage.
    missing_values = df.isnull().sum()
    percentage_missing = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage Missing': percentage_missing})
    print("Missing Values:")
    print(missing_info)
    print('\n\n')
    
    # Explore categorical features, showing unique values for each with their counts.
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        if df[feature].nunique() > 10:
            print(f"Unique values for {feature}:")
            print(df[feature].value_counts()[:5])
            print(f"& many more unique {feature}")
            print("\n")
        else:
            print(f"Unique values for {feature}:")
            print(df[feature].value_counts()[:6])
            print("\n")
    
    # Examine numerical features and display basic statistics like mean, median, standard deviation, skewness, and kurtosis.
    numerical_features = df.select_dtypes(exclude=['object']).columns
    for feature in numerical_features:
        print(f"Statistics for {feature}:")
        print(f"Mean: {df[feature].mean()}")
        print(f"Median: {df[feature].median()}")
        print(f"Standard Deviation: {df[feature].std()}")
        print(f"Skewness: {df[feature].skew()}")
        print(f"Kurtosis: {df[feature].kurtosis()}")
        print("\n")
    
    # Display the correlation matrix of numerical features.
    print("Correlation Matrix:")
    print(df.corr(numeric_only=True))



def Nullvalues(df,target, high_threshold=0.4, low_threshold=0.05):
    if df is None or df.empty:
        raise ValueError("Input DataFrame is either None or empty")
    
    
    actions = []

    try:
        cols_to_drop = [col for col in df.columns if df[col].nunique() == len(df)]
        for col in cols_to_drop:
            if col==target:
                continue
            actions.append(f"Dropped column '{col}' because it contains all unique values")
            df = df.drop(col, axis=1)

        for col in df.columns:
            null_percentage = df[col].isnull().sum() / len(df)

            if null_percentage >= high_threshold:
                actions.append(f"Dropped column '{col}' (null percentage: {null_percentage:.2%})")
                df = df.drop(col, axis=1)

            elif low_threshold <= null_percentage < high_threshold:
                if df[col].dtype == 'O':
                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                    actions.append(f"Filled null values in '{col}' with mode (null percentage: {null_percentage:.2%})")
                else:
                    if abs(df[col].skew()) > 0.5:
                        df[col].fillna(df[col].median(), inplace=True)
                        actions.append(f"Filled null values in '{col}' with median (null percentage: {null_percentage:.2%})")
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)
                        actions.append(f"Filled null values in '{col}' with mean (null percentage: {null_percentage:.2%})")

            elif null_percentage > 0 and null_percentage <= low_threshold:
                df.dropna(subset=[col], inplace=True)
                actions.append(f"Dropped rows with missing values from '{col}' (containing null percentage: {null_percentage:.2%})")

        # original_columns = df.columns.tolist()
        # df.columns = df.columns.str.replace(r'[\\\s\/]', '_', regex=True)

        # for original_col, new_col in zip(original_columns, df.columns):
        #     if original_col != new_col:
        #         actions.append(f"Renamed column '{original_col}' to '{new_col}'")

        def drop_columns_by_majority_value(df,target, threshold=0.8):
            columns_to_drop = []

            for column in df.columns:
                # Find the most common value in the column
                most_common_value_count = df[column].value_counts().max()
                unique_ratio = most_common_value_count / len(df[column])
                
                if unique_ratio >= threshold:
                    columns_to_drop.append(column)

            if columns_to_drop:
                if target in columns_to_drop:
                    columns_to_drop.pop(target)
                df = df.drop(columns=columns_to_drop)
                print(f"Dropped columns with the majority value occurring more than {threshold*100}% of the time: {columns_to_drop}")
            else:
                print("No columns with the majority value occurring more than 80% of the time found in the DataFrame.")

            return df

        df = drop_columns_by_majority_value(df)

        for column in df.select_dtypes(include=['object']):
            df[column] = df[column].apply(lambda x: x.lower() if isinstance(x, str) else x)

        def calculate_vif(dataframe):
            numeric_data = dataframe.select_dtypes(include=['number'])
            if target in numeric_data:
                numeric_data.drop(target,inplace=True) 
            vif_data = pd.DataFrame()
            vif_data["Feature"] = numeric_data.columns
            vif_data["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]
            return vif_data

        vif_data = calculate_vif(df)

        vif_threshold = 10

        high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]

        if not high_vif_features.empty:
            feature_to_drop = high_vif_features.iloc[0]["Feature"]
            df = df.drop(columns=feature_to_drop)
            actions.append(f"Dropped feature '{feature_to_drop}' due to high VIF")
        else:
            actions.append("No high VIF features found.")

        df.drop_duplicates(inplace=True)
        actions.append(f"Dropped duplicate rows")

        for action in actions:
            print("\n")
            print(action)

        num_rows, num_columns = df.shape
        print(f"\nNumber of rows remaining: {num_rows}")
        print(f"Number of columns remaining: {num_columns}")

    except Exception as e:
        # print(f"An error occurred: {str(e)}")
        logging.error(f"An Error occurred: {str(e)}")
        
    return df

def find_lat_lon_columns(dataframe, lat_range=(-90, 90), lon_range=(-180, 180)):
    lat_patterns = ['lat', 'latitudinal', 'y', 'latitude_deg', 'latit', 'lat_deg', 'latitude_degrees', 'latitude',"northing"]
    lon_patterns = ['lon', 'longitudinal', 'x', 'longitude_deg', 'long', 'lon_deg', 'longitude_degrees', 'longitude',"easting"]

    lat_lon_columns = []

    for column in dataframe.columns:
        # Check if the column name contains latitude patterns
        if any(pattern in column.lower() for pattern in lat_patterns):
            # Check if the values in the column are within the latitude range
            if dataframe[column].dtype in ['float64', 'float32'] and dataframe[column].between(lat_range[0], lat_range[1]).all():
                lat_lon_columns.append(column)
        # Check if the column name contains longitude patterns
        elif any(pattern in column.lower() for pattern in lon_patterns):
            # Check if the values in the column are within the longitude range
            if dataframe[column].dtype in ['float64', 'float32'] and dataframe[column].between(lon_range[0], lon_range[1]).all():
                lat_lon_columns.append(column)

    if len(lat_lon_columns) == 2:
        return tuple(lat_lon_columns)
    else:
        return None
    
# plot on world map using latitude and longitude column if exist
def plot_using_lat_lon_columns(result, df, target):
        
    # Function to get city name based on latitude and longitude
    def get_city_name(latitude, longitude):
        cities = []
        for city_data in gc.get_cities().values():
            city_latitude = city_data['latitude']
            city_longitude = city_data['longitude']
            distance = ((latitude - city_latitude) ** 2 + (longitude - city_longitude) ** 2) ** 0.5
            cities.append((city_data['name'], distance))

        # Sort cities by distance and return the closest city name
        closest_city = min(cities, key=lambda x: x[1])
        return closest_city[0]

    try:

        if result:
            print("Latitude and longitude columns found:", result)
            lat_col, lon_col = result
            # Initialize geonamescache
            gc = GeonamesCache()
            
            # Create a new DataFrame with city names, latitudes, and longitudes
            new_data = {'Latitude': df[lat_col],
                        'Longitude': df[lon_col]}
            new_data['City'] = [get_city_name(lat, lon) for lat, lon in zip(df['Latitude'], df['Longitude'])]
            map_df = pd.DataFrame(new_data)

            df['City_lat_lon'] = [get_city_name(lat, lon) for lat, lon in zip(df[lat_col], df[lon_col])]

            maps_filepath = os.path.join(plot_filepath,"maps")
            os.mkdir(maps_filepath)
            fig = px.scatter_mapbox(df, lat=df[lat_col], lon=df[lon_col], hover_name=df['City_lat_lon'], color=target, size_max=15, zoom=2, mapbox_style='carto-positron')
            
            # fig.show()
            fig.write_html(os.path.join(maps_filepath,"scatter_mapbox.html")) 
            df.drop("City_lat_lon", axis = 1, inplace=True)
        else:
            print("Latitude and longitude columns not found.")
            
    except Exception as e:
        print(f"Error generating plot: {e}")


def plot_on_world_map(df, target):
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        print("Input DataFrame is either None, empty, or not a DataFrame.")
        logging.info("Input DataFrame is either None, empty, or not a DataFrame.")
        return
    
    if target not in df.columns:
        print("Target column not found in DataFrame.")
        logging.info("Target column not found in DataFrame.")
        return

    def detect_geospatial_column(column):
        if not pd.api.types.is_string_dtype(column):
            return "none"
        
        gc = geonamescache.GeonamesCache()
        country_names = set(gc.get_countries_by_names().keys())
        city_names = set(city['name'] for city in gc.get_cities().values())

        city_count = sum(1 for x in column if x in city_names)
        country_count = sum(1 for x in column if x in country_names)

        if city_count >= country_count:
            return "city"
        elif country_count > city_count:
            return "country"
        else:
            return "none"
 
    def get_city_lat_lng(city_name):
        city_info = gc.get_cities_by_name(city_name)
        if city_info:
            city_info = list(city_info[0].values())[0]
            return city_info.get('latitude'), city_info.get('longitude')
        return None, None

    def get_country_lat_lng(country_name):
        country = pycountry.countries.get(name=country_name)
        if not country:
            print(f"No match found for country: {country_name}")
            return None, None

        try:
            alpha2 = country.alpha_2
            country_info = gc.get_countries()[alpha2]
            if country_info:
                capital = country_info['capital']
                capital_info_list = gc.get_cities_by_name(capital)
                for capital_info in capital_info_list:
                    for city_id, city_data in capital_info.items():
                        if city_data['countrycode'] == alpha2:
                            return float(city_data['latitude']), float(city_data['longitude'])
                print(f"No geonamescache data found for the capital of {country_name}")
                return None, None
            else:
                print(f"No geonamescache data found for country: {country_name}")
                return None, None
        except Exception as e:
            logging.error(f"Error retrieving coordinates for {country_name}: {e}")
            return None, None

    col_name = None
    for col in df.columns:
        result = detect_geospatial_column(df[col])
        if result != "none":
            col_name = col
            geo_type = result
            break

    if not col_name:
        print("No geospatial column found.")
        return

    gc = geonamescache.GeonamesCache()

    if geo_type == "city":
        df['Latitude'], df['Longitude'] = zip(*df[col_name].apply(get_city_lat_lng))
    elif geo_type == "country":
        df['Latitude'], df['Longitude'] = zip(*df[col_name].apply(get_country_lat_lng))

    df = df.dropna(subset=['Latitude', 'Longitude'])

    try:
        maps_filepath = os.path.join(plot_filepath,"maps")
        os.mkdir(maps_filepath)
        fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', hover_name=col_name, color=target, size_max=15, zoom=2, mapbox_style='carto-positron')
        # fig.show()
        fig.write_html(os.path.join(maps_filepath,"scatter_mapbox.html")) 
    except Exception as e:
        logging.error(f"Error generating plot: {e}")




def visualize_date_with_target(df, target_column):
    if df.empty:
        print("DataFrame is empty.")
        return

    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in DataFrame.")
        return
    
    # Define the date regex pattern
    date_regex_pattern = r"(\d{4}[-/ ,]\d{2}[-/ ,]\d{4}|\d{2}[-/ ,]\d{2}[-/ ,]\d{4}|\d{4}-\d{2}-\d{2})\s?\d{2}:\d{2}:\d{2}|\d{2,4}[- /,]\d{2}[- /,]\d{2,4}"

    # Find columns that match the provided regex pattern
    matching_columns = [col for col in df.columns if df[col].apply(str).str.match(date_regex_pattern, na=False).any()]

    if not matching_columns:
        print("No columns matching the date pattern found in the DataFrame.")
        return
    
    # Initialize date_column to None
    date_column = None

    # Iterate through matching columns to find the first datetime column
    for col in matching_columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_column = col
            break

    # If no datetime column is found, use the first matching column
    if date_column is None:
        date_column = matching_columns[0]

    try:
        other_filepath = os.path.join(plot_filepath,"other")
        os.mkdir(other_filepath)
        # If the column is not already datetime, convert it to datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Check if there are at least 50 rows
        if len(df) < 50:
            print("DataFrame has less than 50 rows. Plotting all available data.")
            df_last = df
        else:
            df_last = df[-50:]

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_last, x=date_column, y=target_column)
        plt.title(f"{date_column} vs. {target_column}")
        plt.xlabel(date_column)
        plt.ylabel(target_column)
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(other_filepath,"date_with_target_plot.png"))
    except Exception as e:
        logging.error(f"An error occurred: {e}")



def integrated_outlier_visualization(df,target, zscore_threshold=3):

    
    def handle_outliers(df, col, zscore_threshold=3):
        # Calculate Z-scores
        if df[col].nunique()>20:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            zscore_outliers = (z_scores > zscore_threshold)
        
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

            # Decide which method to use for this column
            if zscore_outliers.sum() < iqr_outliers.sum():
                print(f"Removed outliers in '{col}' using Z-score method.")
                df = df[~zscore_outliers]
            else:
                print(f"Removed outliers in '{col}' using IQR method.")
                df = df[~iqr_outliers]
        
        return df

    # Select numerical columns (int and float) from the DataFrame.
    num_cols = df.select_dtypes(include=['int','float']).columns

    if target in num_cols:
        num_cols = num_cols.drop(target)

    # Creating file for storing outlier plots
    A_outlier_filepath = os.path.join(plot_filepath,"After")
    B_outlier_filepath = os.path.join(plot_filepath,"Before")
    os.mkdir(A_outlier_filepath)
    os.mkdir(B_outlier_filepath)
    # Create a figure with plotly interactive charts for box plots and violin plots of numerical columns.
    for i in num_cols:
        if df[i].nunique()>10:
            # Display box plot

            fig = px.box(df, y=df[i], title=f'Box Plot for {i}')
            # fig.show()

            fig.write_html(os.path.join(B_outlier_filepath,f"{str(i)}_box_plot.html"))

            # Display violin plot
            fig = px.violin(df, y=df[i], box=True, points="all", title=f'Violin Plot for {i}')
            # fig.show()
            fig.write_html(os.path.join(B_outlier_filepath,f"{str(i)}_violin_plot.html"))

    # Ask the user if they want to remove outliers for all columns
    remove_outliers = input("Do you want to remove outliers for all columns? (yes/no): ")

    if remove_outliers.lower() == 'yes':
        for i in num_cols:
            if df[i].nunique()>10:
                
                df = handle_outliers(df, i, zscore_threshold)
            
                # Show updated box plot after removing outliers
                fig = px.box(df, y=df[i], title=f'Box Plot for {i} (After Removing Outliers)')
                # fig.show()
                fig.write_html(os.path.join(A_outlier_filepath,f"{str(i)}_box_plot.html"))

                # Show updated violin plot after removing outliers
                fig = px.violin(df, y=df[i], box=True, points="all", title=f'Violin Plot for {i} (After Removing Outliers)')
                # fig.show()
                fig.write_html(os.path.join(A_outlier_filepath,f"{str(i)}_violin_plot.html"))

    return df




def data_distribution_Viz(df):
    # Check if DataFrame is empty
    if df.empty:
        print("The DataFrame is empty.")
        return
    
    # Select numerical columns (int and float) from the DataFrame.
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    
    # Check if there are no numeric columns
    if len(num_cols) == 0:
        print("The DataFrame does not have any numeric columns.")
        return
    
    # Determine the number of subplots per row in the figures.
    f = len(num_cols) / 2
    
    try:
        data_dist_filepath = os.path.join(plot_filepath,"data_dist")
        os.mkdir(data_dist_filepath)
        # Create a figure with subplots for dist plots of numerical columns.
        plt.figure(figsize=(10, 8))
        count = 1
        for i in num_cols:
            plt.subplot(int(f) + 1, 3, count)
            sns.distplot(df[i].dropna(), kde=True)  # Drop NaN values and plot
            plt.title(f'Distribution of {i}')
            count += 1
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(data_dist_filepath,"dist_num_subplot.png"))
        
        # Create a figure with subplots for hist plots of numerical columns.
        plt.figure(figsize=(10, 8))
        count = 1
        for i in num_cols:
            plt.subplot(int(f) + 1, 3, count)
            sns.histplot(df[i].dropna(), bins=30)  # Drop NaN values and plot
            plt.title(f'Histogram of {i}')
            count += 1
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(data_dist_filepath,"hist_num_subplots.png"))
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")




def univariate_analysis(df):
    
    # Check if DataFrame is empty
    if df.empty:
        print("The DataFrame is empty.")
        return
    
    # Select numerical (int64 and float) and categorical (object) columns from the DataFrame.
    num_cols = df.select_dtypes(include=['int64', 'float']).columns
    cat_cols = df.select_dtypes(include='object').columns
    
    # Check if there are no numeric and categorical columns
    if len(num_cols) == 0 and len(cat_cols) == 0:
        print("The DataFrame does not have any numeric or categorical columns.")
        return
    
    try:
        univar_filepath = os.path.join(plot_filepath,"univariate")
        os.mkdir(univar_filepath)
        # Calculate the number of subplots per row for numerical and categorical columns.
        f1 = len(num_cols) / 2
        
        # Create a figure for Histogram of numerical columns.
        if len(num_cols) > 0:
            fig_hist = make_subplots(rows=int(f1) + 1, cols=3, subplot_titles=[f'Hist Plot for {i}' for i in num_cols])
            count = 0
            for i in num_cols:
                hist_fig = go.Histogram(x=df[i].dropna(), name=f'Histogram for {i}')
                fig_hist.add_trace(hist_fig, row=int(count / 3) + 1, col=(count % 3) + 1)
                count += 1
                
            fig_hist.update_layout(title="Hist Plot for Numerical columns", width=1250, height=1050)
            # fig_hist.show()
            fig_hist.write_html(os.path.join(univar_filepath,"hist_subplot.html"))
        else:
            print("No numeric columns found in the DataFrame.")
            
        # Create a figure for pie chart of categorical columns.
        if len(cat_cols) > 0:
            for i in cat_cols:
                if df[i].nunique() <= 10:
                    pie_fig = px.pie(df, names=df[i], title=f'Pie Chart for "{i}" column')
                    # pie_fig.show()
                    pie_fig.write_html(os.path.join(univar_filepath,str(i)+"_pie_chart.html"))
                
                else:
                    print(f'The categorical column "{i}" has more than 10 unique values, skipping pie chart.')
        else:
            print("No categorical columns found in the DataFrame.")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")






def corr_with_target(df,target):

    try:
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("The DataFrame is empty.")

        # Check if target exists
        if target not in df.columns:
            raise ValueError("Target column not found in DataFrame.")
    
        # Select numerical and categorical columns from the DataFrame.
        num_cols = df.select_dtypes(include=['int64', 'float64'])
        cat_cols = df.select_dtypes(include='O')       

        # Remove target from num_cols and cat_cols if exists
        num_cols = num_cols.drop(columns=target, errors='ignore')
        cat_cols = cat_cols.drop(columns=target, errors='ignore')

        corr_filepath = os.path.join(plot_filepath,"corr")
        os.mkdir(corr_filepath)

        # Create figures based on target if target is categorical
        if df[target].dtypes == 'O':
            # Crosstab for categorical columns
            for i in cat_cols:
                if df[i].nunique() > 1 and df[i].nunique() < 7:
                    cross_tab = pd.crosstab(index=df[i], columns=df[target])
                    fig = px.bar(cross_tab, barmode='group')
                    fig.update_layout(title=f'Crosstab for {i} and {target}')
                    # fig.show()
                    fig.write_html(os.path.join(corr_filepath,f"{str(i)}_categorical_target.html"))

            # Barplot for numerical columns
            for j in num_cols:
                if df[j].nunique() > 1 and df[j].nunique() < 7:
                    fig = px.bar(df, x=j, y=target, labels={target: "Percentage"}, title=f'Barplot for {j} vs {target}')
                    # fig.show()
                    fig.write_html(os.path.join(corr_filepath,f"{str(j)}_numerical_target_barplot.html"))

            # Countplot for numerical columns
            for j in num_cols:
                if df[j].nunique() > 1 and df[j].nunique() < 10:
                    fig = px.bar(df, x=j, color=target, labels={target: "Percentage"},
                                 title=f'Countplot for {j} vs {target}')
                    # fig.show()
                    fig.write_html(os.path.join(corr_filepath,f"{str(j)}_numerical_countplot.html"))

            # Pairplot
            if len(df.columns) > 1:
                sns.set(style="ticks")
                sns.pairplot(df, hue=target, palette="husl", markers=["o", "s", "D"])
                # plt.show()
                plt.savefig(os.path.join(corr_filepath,"correlation_pairplot.png"))

        else:
            # Scatterplot for numerical columns
            for i in num_cols:
                if df[i].nunique() > 1:
                    sns.scatterplot(x=df[i], y=target, hue=target, data=df)
                    # plt.show()
                    plt.savefig(os.path.join(corr_filepath,f"{str(i)}_numeric_scatter.png"))

            # Correlation heatmap
            corr_matrix = df.corr(numeric_only=True)
            fig = px.imshow(corr_matrix)
            fig.update_layout(title="Correlation Heatmap")
            # fig.show()
            fig.write_html(os.path.join(corr_filepath,"corr_heatmap.html"))

            # Pairplot
            # if len(df.columns) > 1:
            #     # Specify other variables for the pairplot
            #     other_vars = [col for col in df.columns if col != target]

            #     # Check if there are variables for grid columns
            #     if not other_vars:
            #         raise ValueError("No variables found for grid columns.")

            #     sns.set(style="ticks")
            #     sns.pairplot(df, hue=target, palette="husl", markers=["o", "s", "D"])
            #     plt.show()
                # plt.savefig(os.path.join(corr_filepath,"corr_pairplot.png"))

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

                

def remove_low_correlation_features(df, target, threshold=0.05):
    
    try:
        # Check if DataFrame is empty
        if df.empty:
            print("The DataFrame is empty.")
            return df

        # Check if target exists
        if target not in df.columns:
            print("Target column not found in DataFrame.")
            return df

        # Ensure the target is numerical
        if not pd.api.types.is_numeric_dtype(df[target]):
            print("Target column is not numerical. Correlation calculation is not possible.")
            return df

        # Ensure threshold is in the valid range
        if not (0 <= threshold <= 1):
            print("Threshold must be between 0 and 1.")
            return df

        # Calculate the correlation between features and the target
        correlations = df.corr(numeric_only=True).get(target)

        # Check if there are numerical features to correlate
        if correlations is None:
            print("No numerical features found for correlation calculation.")
            return df

        # Exclude target
        correlations = correlations.drop(target, errors='ignore')

        # Identify features with low or no correlation with the target
        low_correlation_features = correlations[(correlations.abs() < threshold) | (correlations.isna())].index.tolist()

        # Check if there are any low or no correlation features to be removed
        if not low_correlation_features:
            print("No low or no correlation features found.")
            return df

        # Display and ask for user permission to remove low or no correlated features
        print(f"Features with low or no correlation: {', '.join(low_correlation_features)}")
        response = input("Do you want to remove these features? (yes/no): ")

        if response.lower() == 'yes':
            df.drop(columns=low_correlation_features, inplace=True)
            print(f"Removed features: {', '.join(low_correlation_features)}")
        else:
            print("No features removed.")

        return df
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

def encode_training_data(df, target, encoding_type='ordinal'):
    # Check if target column exists in DataFrame
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # Check if all columns except target are categorical
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
    if len(categorical_columns) == (df.shape[1] - 1):
        return df, {}, {}, None  # Return the original DataFrame as is

    # global label_encodings
    # global feature_encodings
    # global ohe_columns

    label_encodings_col = {}
    feature_encodings_col = {}
    ohe_col = None

    X = df.drop(target, axis=1)

    # Encoding target column if it's of type object
    if df[target].dtype == 'O':
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        label_encodings_col[target] = le

    # Encoding features2

    for column in X:
        if df[column].dtype == 'O' or df[column].dtype == 'bool':
            if encoding_type == 'ordinal':
                oe = OrdinalEncoder()
                df[column] = oe.fit_transform(df[[column]])
                feature_encodings_col[column] = oe
            elif encoding_type == 'onehot':
                ohe = OneHotEncoder(drop='first')
                ohe_df = ohe.fit_transform(df[[column]])
                ohe_df = pd.DataFrame(ohe_df.toarray(), columns=[f"{column}_{c}" for c in ohe.get_feature_names_out([column])][1:])
                df = pd.concat([df, ohe_df], axis=1)
                df.drop(column, axis=1, inplace=True)
                if ohe_col is None:
                    ohe_col = ohe_df.columns
                else:
                    ohe_col = ohe_col.append(ohe_df.columns)

    return df, label_encodings_col, feature_encodings_col, ohe_col




def encode_new_data(new_data, le_es, fe_es, ohe_cls):

    # print('LE:',le_es, 'FE', fe_es, 'OHE', ohe_cls)

    for column in new_data.columns:
        if column in le_es:
            le = le_es[column]
            new_data[column] = le.transform(new_data[column])
        if column in fe_es:
            oe = fe_es[column]
            known_categories = oe.categories_[0]


            if type(known_categories[0])==bool:
                new_data[column] = new_data[column].apply(lambda x: x if x in known_categories else False)
            elif type(known_categories[0])=='O':
                new_data[column] = new_data[column].apply(lambda x: x if x in known_categories else "Unknown")
            # print('\n After New Data', new_data[column])
            new_data[column] = oe.transform(new_data[[column]])

            
        if ohe_cls is not None and any(ohe_col.startswith(column) for ohe_col in ohe_cls):
            known_categories = [ohe_col.replace(f"{column}_", "") for ohe_col in ohe_cls if ohe_col.startswith(column)]
            for category in new_data[column].unique():
                if category not in known_categories:
                    # print(category,'not in ',column)
                    raise ValueError(f"Unknown category '{category}' in column '{column}'.")
            ohe = OneHotEncoder(drop='first', categories=[known_categories])
            ohe_df = ohe.fit_transform(new_data[[column]])
            ohe_df = pd.DataFrame(ohe_df.toarray(), columns=[f"{column}_{c}" for c in ohe.get_feature_names_out([column])][1:])
            new_data = pd.concat([new_data, ohe_df], axis=1)
            new_data.drop(column, axis=1, inplace=True)
    return new_data
 
def automatic_sampling(df, target_column):
 
    # Separate the minority and majority classes
    minority_class = df[target_column].value_counts().idxmin()
    minority_df = df[df[target_column] == minority_class]
    majority_df = df[df[target_column] != minority_class]
 
    
    methods = ['oversample', 'undersample', 'smote']
 
    best_method = None
    best_score = 0
    balanced_data = None
 
    # Determine the type of classification task (binary or multiclass)
    num_classes = df[target_column].nunique()
    if num_classes == 2:
        scoring = 'roc_auc'  # For binary classification
    else:
        scoring = 'f1_macro'  # For multiclass classification
 
    for method in methods:
        if method == 'oversample':
            # Oversample the minority class
            oversampled_minority = resample(minority_df, replace=True, n_samples=len(majority_df), random_state=142)
            balanced_df = pd.concat([majority_df, oversampled_minority])
 
            
        elif method == 'undersample':
            # Undersample the majority class
            undersampled_majority = resample(majority_df, replace=False, n_samples=len(minority_df), random_state=142)
            balanced_df = pd.concat([minority_df, undersampled_majority])
 
            
#         elif method == 'smote':
#             # Use SMOTE (Synthetic Minority Over-sampling Technique)
#             X = majority_df.drop(target_column, axis=1)
#             y = majority_df[target_column]
#             smote = SMOTE(random_state=142)
#             X_resampled, y_resampled = smote.fit_resample(X, y)
#             balanced_df = pd.concat([minority_df, pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1)
 
            
        # Evaluate the method using cross-validation
        X_balanced = balanced_df.drop(target_column, axis=1)
        y_balanced = balanced_df[target_column]
 
        if scoring == 'roc_auc':
            scorer = make_scorer(roc_auc_score)
        else:
            scorer = make_scorer(f1_score, average='macro')
 
        scores = cross_val_score(RandomForestClassifier(), X_balanced, y_balanced, cv=5, scoring=scorer)
 
        # Keep track of the best method
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_method = method
            balanced_data = balanced_df.copy()
 
    return balanced_data, best_method



def scaling(df, target, new_data=None):

    X = df.drop(target, axis=1)
    y = df[target]

    global fit_scaler
    def fit_scaler(X):
        scalers = {}
        for column in X.columns:
            # Check the data type of the column and select the appropriate scaler
            if X[column].dtype in ['int64', 'float64']:
                # Check skewness to decide between StandardScaler and MinMaxScaler
                if abs(skew(X[column])) <= 0.5:  # Approximately normal
                    scalers[column] = StandardScaler()
                else:
                    scalers[column] = MinMaxScaler()

                scalers[column].fit(X[column].values.reshape(-1, 1))
        return scalers

        
    global transform_with_scaler
    def transform_with_scaler(X, scalers):
        for column in X.columns:
            if column in scalers:
                X[column] = scalers[column].transform(X[column].values.reshape(-1, 1))
        return X

    if new_data is not None:

        X_train = X
        y_train = y
        X_test = new_data
        y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

    global scalers
    scalers = fit_scaler(X_train)
        
    X_train = transform_with_scaler(X_train, scalers)
    X_test = transform_with_scaler(X_test, scalers)

    return X_train, X_test, y_train, y_test






def Threshold(model, X_train, X_test, y_train, y_test):
    try:
        # Train the provided machine learning model on the training data
        model.fit(X_train, y_train)
        
        # Make predictions on the training and testing datasets
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Check if model has 'predict_proba' method
        if not hasattr(model, "predict_proba"):
            raise AttributeError("The model does not have the 'predict_proba' method.")
        
        # Obtain predicted probabilities for the training and testing datasets
        y_train_proba = model.predict_proba(X_train)[:, -1]
        y_test_proba = model.predict_proba(X_test)[:, -1]
        
        # Calculate and visualize the Precision-Recall (P-R) Curve
        p, r, th = precision_recall_curve(y_train, y_train_proba)
        
        # Create a plot for the P-R curve with precision and recall values
        plt.figure(figsize=(6, 6))
        sns.lineplot(x=th, y=p[:-1], label='precision')
        sns.lineplot(x=th, y=r[:-1], label='Recall')
        plt.title('Precision Recall Curve')
        plt.axvline(0.72)  # Vertical line for a specific threshold
        plt.show()
        
        # Define a function for calculating and displaying classification metrics
        def metrics(y_actual, y_proba, th):
            y_pred_temp = [1 if p > th else 0 for p in y_proba]
            accuracy = accuracy_score(y_actual, y_pred_temp)
            recall = recall_score(y_actual, y_pred_temp, zero_division=0)
            precision = precision_score(y_actual, y_pred_temp, zero_division=0)
            f1 = f1_score(y_actual, y_pred_temp, zero_division=0)
            roc_auc = roc_auc_score(y_actual, y_pred_temp)
            return {"Accuracy": round(accuracy, 2), 'Recall': round(recall, 2), 'Precision': round(precision, 2), 'F1_Score': round(f1, 2), 'ROC_AUC': round(roc_auc, 2)}
        
        # Calculate and display the metrics for both the training and testing data at a specific threshold
        print("Train Data")
        print(metrics(y_train, y_train_proba, 0.72))
        print("Test Data")
        print(metrics(y_test, y_test_proba, 0.72))
        
        # Plot the Receiver Operating Characteristic (ROC) curve
        fpr, tpr, th = roc_curve(y_train, y_train_proba)
        
        # Create a plot for the ROC-AUC curve
        plt.figure(figsize=(6, 6))
        sns.lineplot(x=fpr, y=tpr)
        sns.lineplot(x=[0.0, 1], y=[0.0, 1], color='red', linestyle='--') 
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title('ROC AUC Curve')
        plt.show()
    
    except Exception as e:
        logging.error("An error occurred:", str(e))


def train_and_save_model(X_train, y_train, model_name, mode):
    global automl
    automl = AutoML(verbose=0)
    automl.fit(X_train, y_train, task=mode, time_budget=300)

    with open(model_name, 'wb') as model_file:
        pickle.dump(automl, model_file)

    return automl



def BinaryClassification(df,target):
    
    global df_nn
    # global df_new
    info(df)

    df = Nullvalues(df,target)
    df_nn = df.copy()
    
    print()

    integrated_outlier_visualization(df, target)

    data_distribution_Viz(df)

    # Perform univariate analysis and calculate feature correlations with the target variable.
    univariate_analysis(df)
    corr_with_target(df,target)
    
    # Plotting on world map if lat, lon, city, or country column present
    result = find_lat_lon_columns(df)
    if result != None:
        plot_using_lat_lon_columns(result,df, target)
    else:
        plot_on_world_map(df, target)

    # dataReport(df)
    # VisualAnalysis(df)

    global automl
    global encoded_df
    global label_encodings 
    global feature_encodings
    global ohe_columns


    encoded_df, label_encodings, feature_encodings, ohe_columns = encode_training_data(df_nn, target)

    balanced_data, best_method = automatic_sampling(df_nn,target)
    balanced_data.to_csv("balanced_data_"+str(target)+".csv")
    print(f"Best method to after automatic sampling is {best_method}")

    # Split the data into training and testing sets and perform feature scaling.
    X_train, X_test, y_train, y_test = scaling(encoded_df, target)
        

    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, 'saved_models')

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Generate a model name based on the DataFrame name
    file = Path(filename).stem
    model_name = f"{model_folder}/{file}_{target}_classification_model.pkl"

    # Check if the model file already exists
    if os.path.exists(model_name):
        use_existing_model = input("A trained model already exists. Do you want to use it? (yes/no): ").lower()
        if use_existing_model == 'yes':
            # with open(model_name, 'rb') as model_file:
            automl = pickle.load(open(model_name, 'rb'))
            print("Using the existing model.")
        else:
            automl = train_and_save_model(X_train, y_train, model_name, "classification")
    else:
        automl = train_and_save_model(X_train, y_train, model_name,"classification")

    y_pred_train = automl.predict(X_train)
    y_pred_test = automl.predict(X_test)
        
    print('Train Data')
    print(classification_report(y_train,y_pred_train))
    print('Test Data')
    print(classification_report(y_test,y_pred_test))

    print("\n\n\n")
#         print("Best Model:",automl.best_estimator)
        
    if automl.best_estimator == 'rf':
        Best_Model = 'Random Forest'

    elif automl.best_estimator == 'lgbm':
        Best_Model = 'Light Gradient Boosting'

    elif automl.best_estimator == 'xgboost' or automl.best_estimator == 'xgb_limitdepth':
        Best_Model = 'Extreme Gradient Boosting'

    elif automl.best_estimator == 'catboost':
        Best_Model = 'CatBoost'

    elif automl.best_estimator == 'lrl1':
        Best_Model = 'LRL1'

    elif automl.best_estimator == 'extra_tree':
        Best_Model = 'Extra Tree Classifier'

    else:
        Best_Model = automl.best_estimator
        
    print("Best Model on this data is",Best_Model)
    print("\n\n\n")
    print("Best Parameters:",automl.best_config)
            

    y_train_proba = automl.predict_proba(X_train)[:,-1]
    y_train_proba

    y_test_proba = automl.predict_proba(X_test)[:,-1]
    y_test_proba



def MultiClassClassification(df,target):
    
    global df_nn
        # Display dataset information, check for missing values, visualize outliers and data distribution.
    info(df)
    df = Nullvalues(df,target)
    df_nn = df.copy()
    integrated_outlier_visualization(df, target)
    data_distribution_Viz(df)

    # Perform univariate analysis and calculate feature correlations with the target variable.
    univariate_analysis(df)
    corr_with_target(df, target)

    # Plotting on world map if lat, lon, city, or country column present
    result = find_lat_lon_columns(df)
    if result != None:
        plot_using_lat_lon_columns(result,df, target)
    else:
        plot_on_world_map(df, target)
    # dataReport(df)
    # VisualAnalysis(df)
        
    # Encode categorical features if present in the dataset.

    global automl
    global encoded_df
    global label_encodings 
    global feature_encodings
    global ohe_columns

    encoded_df, label_encodings, feature_encodings, ohe_columns = encode_training_data(df_nn, target)

    # Split the data into training and testing sets and perform feature scaling.
    X_train, X_test, y_train, y_test = scaling(encoded_df, target)


    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, 'saved_models')

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Generate a model name based on the DataFrame name
    file = Path(filename).stem
    model_name = f"{model_folder}/{file}_{target}_classification_model.pkl"

    # Check if the model file already exists
    if os.path.exists(model_name):
        use_existing_model = input("A trained model already exists. Do you want to use it? (yes/no): ").lower()
        if use_existing_model == 'yes':
            # with open(model_name, 'rb') as model_file:
            automl = pickle.load(open(model_name, 'rb'))
            print("Using the existing model.")
        else:
            automl = train_and_save_model(X_train, y_train, model_name, "classification")
    else:
        automl = train_and_save_model(X_train, y_train, model_name, "classification")


    y_pred_train = automl.predict(X_train)
    y_pred_test = automl.predict(X_test)

    print("\n\n\n")
#        print("Best Model:",automl.best_estimator)
        
        
    if automl.best_estimator == 'rf':
        Best_Model = 'Random Forest'

    elif automl.best_estimator == 'lgbm':
        Best_Model = 'Light Gradient Boosting'

    elif automl.best_estimator == 'xgboost' or automl.best_estimator == 'xgb_limitdepth':
        Best_Model = 'Extreme Gradient Boosting'

    elif automl.best_estimator == 'catboost':
        Best_Model = 'CatBoost'

    elif automl.best_estimator == 'lrl1':
        Best_Model = 'LRL1'
            
    elif automl.best_estimator == 'extra_tree':
        Best_Model = 'Extra Tree Classifier'

    else:
        Best_Model = automl.best_estimator
        
    print("Best Model on the data is",Best_Model)
    print("\n\n\n")
    print("Best Parameters:",automl.best_config)

    print("\n\n\n\n\n")
    print('Train Data')
    print(classification_report(y_train,y_pred_train))
    print('Test Data')
    print(classification_report(y_test,y_pred_test))
        


def Regression(df,target):

    info(df)
    global df_new
    df_new = Nullvalues(df,target)
    
    integrated_outlier_visualization(df,target)
    data_distribution_Viz(df)

        # Perform univariate analysis and calculate feature correlations with the target variable.
    univariate_analysis(df)
    corr_with_target(df, target)

    # Plotting on world map if lat, lon, city, or country column present
    result = find_lat_lon_columns(df)
    if result != None:
        plot_using_lat_lon_columns(result,df, target)
    else:
        plot_on_world_map(df, target)

    # dataReport(df)
    # VisualAnalysis(df)
    global df_nn
    df_rlc=remove_low_correlation_features(df_new,target)   
    df_nn = df_rlc.copy()

    df_nn.to_csv(os.path.join('datasets',f'{file}_{target}.csv'), index = False)

    global automl
    global encoded_df
    global label_encodings 
    global feature_encodings
    global ohe_columns

        # Encode categorical features if present in the dataset.
    # global encoded_df,label_encodings, feature_encodings, ohe_columns
    encoded_df, label_encodings, feature_encodings, ohe_columns = encode_training_data(df_nn, target)

        # Split the data into training and testing sets and perform feature scaling.
    X_train, X_test, y_train, y_test = scaling(encoded_df, target)
        
    

    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, 'saved_models')

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # Generate a model name based on the DataFrame name
    model_name = os.path.join(model_folder,f"{file}_{target}_regression_model.pkl")

    # Check if the model file already exists
    if os.path.exists(model_name):
        use_existing_model = input("A trained model already exists. Do you want to use it? (yes/no): ").lower()
        if use_existing_model == 'yes':
            # with open(model_name, 'rb') as model_file:
            automl = pickle.load(open(model_name, 'rb'))
            print("Using the existing model.")
        else:
            automl = train_and_save_model(X_train, y_train, model_name, "regression")
    else:
        automl = train_and_save_model(X_train, y_train, model_name, "regression")

    y_pred_train = automl.predict(X_train)
    y_pred_test = automl.predict(X_test)


    print("\n\n\n")
#         print("Best Model:",automl.best_estimator)
    
        
    if str(automl.best_estimator) == 'rf':
        Best_Model = 'Random Forest'

    elif str(automl.best_estimator) == 'lgbm':
        Best_Model = 'Light Gradient Boosting'

    elif automl.best_estimator == 'xgboost' or automl.best_estimator == 'xgb_limitdepth':
        Best_Model = 'Extreme Gradient Boosting'

    elif automl.best_estimator == 'catboost':
        Best_Model = 'CatBoost'

    elif automl.best_estimator == 'lrl1':
        Best_Model = 'LRL1'
            
    elif automl.best_estimator == 'extra_tree':
        Best_Model = 'Extra Tree Regressor'

    else:
        Best_Model = automl.best_estimator
        
    print("Best Model on the data is",Best_Model)
    print("\n\n\n")
    print("Best Parameters:",automl.best_config)

    print("\n\n\n\n\n")
    print("Train Data")
    print("R2 score:",round(r2_score(y_train,y_pred_train),2))
    print("RMSE Score:",round(np.sqrt(mean_squared_error(y_train,y_pred_train)),2))

    print("\n\n")
    print("Test Data")
    print("R2 Score:",round(r2_score(y_test,y_pred_test),2))
    print("RMSE Score:",round(np.sqrt(mean_squared_error(y_test,y_pred_test)),2))
    print("\n\n\n\n\n")

    
    
    
def AutoModel(df, target):

        
    # Determine the type of machine learning problem based on the number of unique classes in the target variable.
    if df[target].nunique() == 2:
        print('Binary Classification Problem')
        BinaryClassification(df,target)
            
    elif 2 < df[target].nunique() <= 15:
        print('Multi-Class Classification Problem')
        MultiClassClassification(df,target)
            
    else:
        print('Regression Problem')
        Regression(df,target)




def preprocess_data(user_data, df, target, label_encodings, feature_encodings, ohe_columns, scalers):
    # Create a DataFrame with user input.
    input_data = pd.DataFrame([user_data], columns=df.drop(target, axis=1).columns)

    encoded_new_data = encode_new_data(input_data, label_encodings, feature_encodings, ohe_columns)

    # Scale the data
    new_data_scaled = transform_with_scaler(encoded_new_data, scalers)

    return new_data_scaled

def delete_folder(folder_path):
    try:
        # Use shutil.rmtree to delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except FileNotFoundError:  


        logging.info(f"Folder '{folder_path}' not found.")
    except Exception as e:
        logging.info(f"An error occurred: {e}")

def prediction_function(automl, df, target, label_encodings, feature_encodings, ohe_columns, scalers):
    # Get user input for feature values.
    user_input = {}
    for feature in df_nn.columns:
        if feature != target:
            value = input(f"\nEnter the value for '{feature}': ")

    
            if df[feature].dtype == 'int64':
                value = int(value)
            elif df[feature].dtype == 'float64':
                value = float(value)
            elif df[feature].dtype == 'bool':
                value = bool(value)

            user_input[feature] = value

    # Preprocess user input data.
    input_data = preprocess_data(user_input, df, target, label_encodings, feature_encodings, ohe_columns, scalers)

    # Make predictions using the AutoML model.
    encoded_prediction = automl.predict(input_data)[0]

    # Reverse mapping from encoded value to original target value

    # label_encoder = label_encodings[target]
    # original_prediction = label_encoder.inverse_transform([encoded_prediction])[0]

    return encoded_prediction


#START

global filename
filename = r"C:\Users\dheer\OneDrive\Documents\Resume_Project\Auto_ML\datasets\heart.csv"

global target
target = "target"
# print(target)

target = re.sub(r'[\\\s/]', '_', target)
# print(target)

global file
file = Path(filename).stem
global plot_filepath
plot_filepath = os.path.join("plottings",f"{file}_{target}")

if os.path.exists(plot_filepath):
    delete_folder(plot_filepath)


os.mkdir(plot_filepath)

df = pd.read_csv(filename, on_bad_lines='skip')
df.columns = df.columns.str.replace(r"[\\\s\/]", '_', regex=True)
# df.columns = df.columns.str.replace("'","")
AutoModel(df, target)


while True:
    
    next = input('\nYou want to continue predicting?  ')
    
    if next.lower() == 'yes' or next == 'y':
        
        pred = prediction_function(automl,df,target, label_encodings, feature_encodings, ohe_columns, scalers)
        # print(df.sample(1))
        print("\nThe Prediction is ", pred)
    
    else:
        break