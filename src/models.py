import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import ARIMA_PARAMS, SARIMA_PARAMS, ARIMAX_PARAMS
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TimeSeriesModel:
    def __init__(self, df):
        self.df = df

    def split_train_test(self, train_ratio, target_column, exogenous_columns):
        """
        Splits the dataset into train and test sets based on a train-test ratio.
        """
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1.")

        split_size = int(len(self.df) * train_ratio)
        df_train = self.df.iloc[:split_size].copy()
        df_test = self.df.iloc[split_size:].copy()

        y_train = df_train[target_column]
        y_test = df_test[target_column]
        exog_train = df_train[exogenous_columns]
        exog_test = df_test[exogenous_columns]

        print(f"Training Data Size: {len(df_train)}")
        print(f"Testing Data Size: {len(df_test)}")

        return df_train, df_test, y_train, y_test, exog_train, exog_test

    def train_time_series_model(self, model_type, y_train, exog_train=None, **kwargs):
        """
        Trains a time-series model (ARIMA or SARIMAX).
        """
        try:
            if model_type == "ARIMA":
                order = kwargs.get("order", ARIMA_PARAMS["order"])
                model = ARIMA(y_train, order=order)
            elif model_type == "ARIMAX":
                if exog_train is None:
                    raise ValueError(
                        "Exogenous variables (exog_train) must be provided for ARIMAX.")
                order = kwargs.get("order", ARIMAX_PARAMS["order"])
                model = SARIMAX(y_train, order=order, exog=exog_train,
                                enforce_stationarity=False, enforce_invertibility=False)
            elif model_type == "SARIMA":
                order = kwargs.get("order", SARIMA_PARAMS["order"])
                seasonal_order = kwargs.get(
                    "seasonal_order", SARIMA_PARAMS["seasonal_order"])
                model = SARIMAX(y_train, order=order,
                                seasonal_order=seasonal_order)

            model_fit = model.fit()
            print(f"{model_type} Model Summary:\n", model_fit.summary())
            return model_fit
        except Exception as e:
            raise RuntimeError(f"Error training {model_type} model: {e}")

    def predict_time_series_model(self, model_fit, forecast_steps, exog_test=None):
        """
        Predicts future values using a trained time-series model (ARIMA/SARIMAX).
        """
        try:
            if exog_test is not None:
                forecast = model_fit.forecast(
                    steps=forecast_steps, exog=exog_test)
            else:
                forecast = model_fit.forecast(steps=forecast_steps)
            return forecast.to_frame(name="Predicted")
        except Exception as e:
            raise RuntimeError(f"Error during forecasting: {e}")

    def evaluate_model(self, y_test, y_pred):
        """
        Evaluates model performance using RMSE and MAE.
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            # print(f"RMSE: {rmse:.4f}")
            # print(f"MAE: {mae:.4f}")
            return {"RMSE": rmse, "MAE": mae}
        except Exception as e:
            raise RuntimeError(f"Error during model evaluation: {e}")
