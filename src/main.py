import os
from data_loader import load_data
from preprocessing import preprocess_data
from models import TimeSeriesModel
from config import TARGET_COLUMN, TRAIN_RATIO, ARIMA_PARAMS, SARIMA_PARAMS, EXOGENOUS_COLUMNS
from utils import save_evaluation_results, plot_predictions


def main():
    """
    Main pipeline for training, forecasting, and evaluating the ARIMA model.
    """
    print("\nLoading dataset...")
    data = load_data("data/interest_rates.csv")

    print("\nPreprocessing data...")
    df = preprocess_data(data)

    # Initialize the TimeSeriesModel class
    ts_model = TimeSeriesModel(df)

    print("\nSplitting train and test sets...")
    df_train, df_test, y_train, y_test, exog_train, exog_test = ts_model.split_train_test(
        TRAIN_RATIO, TARGET_COLUMN, EXOGENOUS_COLUMNS
    )

    print("\nTraining ARIMA Model...")
    arima_model = ts_model.train_time_series_model(
        "ARIMA", y_train, **ARIMA_PARAMS)

    print("\nTraining SARIMA Model...")
    sarima_model = ts_model.train_time_series_model(
        "SARIMA", y_train, **SARIMA_PARAMS
    )

    print("\nTraining ARIMAX Model...")
    arimax_model = ts_model.train_time_series_model(
        "ARIMAX", y_train, exog_train=exog_train
    )

    print("\nForecasting with ARIMA...")
    y_pred_arima = ts_model.predict_time_series_model(
        arima_model, forecast_steps=len(y_test)
    )

    print("\nForecasting with SARIMA...")
    y_pred_sarimax = ts_model.predict_time_series_model(
        sarima_model, forecast_steps=len(y_test)
    )

    print("\nForecasting with ARIMAX...")
    y_pred_arimax = ts_model.predict_time_series_model(
        arimax_model, forecast_steps=len(y_test), exog_test=exog_test
    )

    print("\nEvaluating ARIMA Model...")
    arima_metrics = ts_model.evaluate_model(y_test, y_pred_arima)

    print("\nEvaluating SARIMA Model...")
    sarima_metrics = ts_model.evaluate_model(y_test, y_pred_sarimax)

    print("\nEvaluating ARIMAX Model...")
    arimax_metrics = ts_model.evaluate_model(y_test, y_pred_arimax)

    print("\nSaving evaluation results...")
    save_evaluation_results(arima_metrics, sarima_metrics, arimax_metrics)

    print("\nSaving prediction plots...")
    plot_predictions(y_test, y_pred_arima, "ARIMA")
    plot_predictions(y_test, y_pred_sarimax, "SARIMA")
    plot_predictions(y_test, y_pred_arimax, "ARIMAX")


if __name__ == "__main__":
    main()
