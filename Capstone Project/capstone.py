"""
ALY 6140 Capstone Project
Written by Jeff Hackmeister

# NYC EMS Incident Dispatch Data available at https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj/about_data
# Filtered for years 2023 - 2025 via online data filter before exporting csv file 
# Data dictionary from NYC Data avilable at https://data.cityofnewyork.us/api/views/76xm-jjuj/files/5f77cf01-4e52-443b-a718-a3e6567e83f2?download=true&filename=EMS_incident_dispatch_data_description.xlsx
"""

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# define customer function for analysis 

# data import 
def load_and_prepare(path):
    """
    :param path: direct file path to the CSV file
    :return: cleaned pandas DataFrame with derived time features and response time column
    """
    COLS = [
        'INCIDENT_DATETIME',
        'BOROUGH',
        'FINAL_CALL_TYPE',
        'INCIDENT_RESPONSE_SECONDS_QY',
        'ZIPCODE'
    ]

    df = pd.read_csv(
        path,
        usecols=COLS,
        dtype={
            'BOROUGH':         'category',
            'FINAL_CALL_TYPE': 'category',
            'ZIPCODE':         'str'
        }
    )

    # Parse timestamp
    df['INCIDENT_DATETIME'] = pd.to_datetime(
        df['INCIDENT_DATETIME'],
        format='%m/%d/%Y %I:%M:%S %p',
        errors='coerce'
    )

    # Derived time features
    df['HOUR']        = df['INCIDENT_DATETIME'].dt.hour.astype('int8')
    df['DAY_OF_WEEK'] = df['INCIDENT_DATETIME'].dt.day_name().astype('category')
    df['MONTH']       = df['INCIDENT_DATETIME'].dt.month.astype('int8')
    df['YEAR']        = df['INCIDENT_DATETIME'].dt.year.astype('int16')

    # Response time in minutes from pre-calculated seconds column
    df['RESPONSE_TIME_MIN'] = pd.to_numeric(
        df['INCIDENT_RESPONSE_SECONDS_QY'], errors='coerce'
    ) / 60

    # Remove outlier response times
    df = df[(df['RESPONSE_TIME_MIN'] > 0) & (df['RESPONSE_TIME_MIN'] <= 120)]

    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df

# missing data report 
def missing_data_report(df):
    """
    :param df: pandas DataFrame containing EMS incident data
    :return: DataFrame showing missing value counts and percentages per column
    """
    report = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_pct':   (df.isna().mean() * 100).round(2)
    }).query('missing_count > 0').sort_values('missing_pct', ascending=False)

    report.plot(kind='barh', y='missing_pct', legend=False, figsize=(10, 6), color='#E63946')
    plt.title('Missing Data by Column')
    plt.xlabel('Missing (%)')
    plt.tight_layout()
    plt.show()

    return report

# response times
def summarise_response_times(df, response_col, group_by=None):
    """
    :param df: pandas DataFrame containing EMS incident data
    :param response_col: column name containing response time values
    :param group_by: optional column to group statistics by (e.g. 'BOROUGH')
    :return: DataFrame of descriptive statistics for response times
    """
    if group_by:
        summary = df.groupby(group_by)[response_col].agg(
            count='count',
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max'
        ).round(2)
    else:
        s = df[response_col].dropna()
        summary = pd.DataFrame([{
            'count':  len(s),
            'mean':   round(s.mean(), 2),
            'median': round(s.median(), 2),
            'std':    round(s.std(), 2),
            'min':    round(s.min(), 2),
            'max':    round(s.max(), 2),
        }])

    df.boxplot(column=response_col, by=group_by, figsize=(10, 6))
    plt.title(f'Response Time by {group_by}' if group_by else 'Response Time Distribution')
    plt.suptitle('')
    plt.xlabel(group_by if group_by else '')
    plt.ylabel('Response Time (min)')
    plt.tight_layout()
    plt.show()

    return summary

# top N call types 
def call_type_profile(df, call_type_col, top_n=15):
    """
    :param df: pandas DataFrame containing EMS incident data
    :param call_type_col: column name containing the call/incident type
    :param top_n: number of top call types to display (default 15)
    :return: DataFrame of top N call types with counts and percentages
    """
    top_types = (
        df[call_type_col]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_types.columns = ['call_type', 'count']
    top_types['percentage'] = (top_types['count'] / len(df) * 100).round(2)

    top_types.plot(
        kind='barh',
        x='call_type',
        y='count',
        legend=False,
        figsize=(10, 6),
        color='#457B9D'
    )
    plt.title(f'Top {top_n} EMS Call Types')
    plt.xlabel('Incident Count')
    plt.ylabel('Call Type')
    plt.tight_layout()
    plt.show()

    return top_types

# temporal heatmap 
def temporal_demand_heatmap(df, hour_col, day_col):
    """
    :param df: pandas DataFrame containing EMS incident data
    :param hour_col: column name containing hour of day (0-23)
    :param day_col: column name containing day of week
    :return: pivot table of incident counts by hour and day
    """
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday']

    pivot = (
        df.groupby([hour_col, day_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=day_order)
    )

    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3)
    plt.title('Incident Volume by Hour and Day')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.show()

    return pivot


# Modeling 
# Model 1 - Response Time Regression 

def response_time_regression(df,
                              response_col='RESPONSE_TIME_MIN',
                              features=['BOROUGH', 'HOUR', 'DAY_OF_WEEK', 'FINAL_CALL_TYPE']):
    """
    :param df: cleaned pandas DataFrame
    :param response_col: column name for response time (target variable)
    :param features: list of feature column names to use as predictors
    :return: trained model, test set actual values, test set predictions
    """
    # Drop rows with missing values in relevant columns
    model_df = df[features + [response_col]].dropna().copy()
 
    # Log-transform response time to correct for right skew
    model_df['LOG_RESPONSE_TIME'] = np.log1p(model_df[response_col])
 
    # Encode categorical features
    encoders = {}
    for col in features:
        if model_df[col].dtype.name in ['category', 'object']:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            encoders[col] = le
 
    X = model_df[features]
    y = model_df['LOG_RESPONSE_TIME']
 
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    # Metrics (back-transformed from log)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}  |  R²: {r2:.4f}")
    print("\nFeature Coefficients:")
    for feat, coef in zip(features, model.coef_):
        print(f"  {feat:<25} {coef:.4f}")
 
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test[:2000], y_pred[:2000], alpha=0.3, s=10, color='#457B9D')
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color='red', linewidth=1.5, linestyle='--', label='Perfect fit')
    ax.set_title('Actual vs Predicted Log(Response Time)')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
    plt.tight_layout()
    plt.show()
 
    return model, y_test, y_pred

# Model 2 - Kmeans clustering for resource allocation 
def kmeans_resource_clusters(df,
                              borough_col='BOROUGH',
                              hour_col='HOUR',
                              response_col='RESPONSE_TIME_MIN',
                              n_clusters=5):
    """
    :param df: cleaned pandas DataFrame
    :param borough_col: column name for borough
    :param hour_col: column name for hour of day
    :param response_col: column name for response time
    :param n_clusters: number of clusters (default 5, one per borough)
    :return: DataFrame with cluster labels, fitted KMeans model
    """
    # Build feature matrix: avg response time and incident volume by borough + hour
    cluster_df = (
        df.groupby([borough_col, hour_col])
        .agg(
            incident_count=(response_col, 'count'),
            avg_response=(response_col, 'mean')
        )
        .reset_index()
        .dropna()
    )
 
    # Normalize features
    features = ['incident_count', 'avg_response', hour_col]
    X = cluster_df[features].copy()
    X_norm = (X - X.mean()) / X.std()
 
    # Elbow plot to validate n_clusters choice
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_norm)
        inertias.append(km.inertia_)
 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_range, inertias, marker='o', color='#457B9D')
    ax.set_title('Elbow Plot — Optimal Number of Clusters')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    plt.tight_layout()
    plt.show()
 
    # Fit final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['CLUSTER'] = kmeans.fit_predict(X_norm)
 
    # Cluster profile summary
    summary = (
        cluster_df.groupby('CLUSTER')
        .agg(
            avg_incidents=('incident_count', 'mean'),
            avg_response=('avg_response', 'mean'),
            peak_hour=(hour_col, lambda x: x.value_counts().idxmax())
        )
        .round(2)
    )
    print("\nCluster Profiles:")
    print(summary.to_string())
 
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        cluster_df[hour_col],
        cluster_df['avg_response'],
        c=cluster_df['CLUSTER'],
        cmap='tab10',
        s=cluster_df['incident_count'] / cluster_df['incident_count'].max() * 200,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('K-Means Clusters: Hour vs Avg Response Time\n(bubble size = incident volume)')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Response Time (min)')
    plt.tight_layout()
    plt.show()
 
    return cluster_df, kmeans

# Model 3 Kernel Density 
def kde_hotspots(df,
                 response_col='RESPONSE_TIME_MIN',
                 borough_col='BOROUGH',
                 hour_col='HOUR',
                 severity_threshold=10):
    """
    :param df: cleaned pandas DataFrame
    :param response_col: column name for response time
    :param borough_col: column name for borough
    :param hour_col: column name for hour of day
    :param severity_threshold: response time (min) above which an incident
                               is classified as high severity (default 10)
    :return: DataFrame with severity labels and KDE values
    """
    kde_df = df[[response_col, borough_col, hour_col]].dropna().copy()
 
    # Classify severity based on response time threshold
    kde_df['SEVERITY'] = np.where(
        kde_df[response_col] >= severity_threshold, 'High', 'Normal'
    )
 
    # KDE plot of response time by severity
    fig, ax = plt.subplots(figsize=(10, 5))
    for severity, color in [('Normal', '#457B9D'), ('High', '#E63946')]:
        subset = kde_df.loc[kde_df['SEVERITY'] == severity, response_col]
        subset.plot.kde(ax=ax, label=severity, color=color, linewidth=2)
    ax.set_title('KDE of Response Time by Severity')
    ax.set_xlabel('Response Time (min)')
    ax.set_xlim(0, 40)
    ax.legend(title='Severity')
    plt.tight_layout()
    plt.show()
 
    # Heatmap: high-severity incident volume by borough and hour
    high_severity = kde_df[kde_df['SEVERITY'] == 'High']
    pivot = (
        high_severity.groupby([borough_col, hour_col])
        .size()
        .unstack(fill_value=0)
    )
 
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, ax=ax)
    ax.set_title(f'High Severity Incidents (response > {severity_threshold} min)\nby Borough and Hour')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Borough')
    plt.tight_layout()
    plt.show()
 
    print(f"\nSeverity breakdown:")
    print(kde_df['SEVERITY'].value_counts().to_string())
    print(f"\nHigh severity by borough:")
    print(high_severity[borough_col].value_counts().to_string())
 
    return kde_df