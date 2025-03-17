import pandas as pd


def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    :param file_path: str, path to the dataset file
    :return: pandas DataFrame
    """

    try:
        df = pd.read_csv(file_path)

        required_columns = ['Year', 'Month', 'Day',
                            'Effective Federal Funds Rate', 'Inflation Rate', 'Unemployment Rate']
        df = df[required_columns]

        return df

    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
