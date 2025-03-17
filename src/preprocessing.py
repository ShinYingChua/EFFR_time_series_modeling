import pandas as pd


def preprocess_data(df):
    """
    Preprocesses the interest rates dataset by:
    - Converting 'Year', 'Month', 'Day' columns into a datetime index.
    - Selecting relevant columns: 'Effective Federal Funds Rate', 'Inflation Rate', 'Unemployment Rate'.
    - Dropping rows with missing values.
    - Renaming columns for consistency.

    :param df: pandas DataFrame, raw dataset loaded from CSV.
    :return: pandas DataFrame with preprocessed data.
    """
    df = df.copy()

    # Convert to datetime format
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Set Date as index
    df.set_index('Date', inplace=True)

    # Select only relevant columns and drop missing values
    df = df[['Effective Federal Funds Rate',
             'Inflation Rate', 'Unemployment Rate']].dropna()

    # Rename columns for better readability
    df.rename(columns={'Effective Federal Funds Rate': 'Interest_Rate',
                       'Inflation Rate': 'Inflation_Rate',
                       'Unemployment Rate': 'Unemployment_Rate'}, inplace=True)

    # Filter data for the last 10 years
    df_10y = df[df.index >= df.index.max() - pd.DateOffset(years=10)]

    return df_10y
