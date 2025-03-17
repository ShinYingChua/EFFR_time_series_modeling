import matplotlib.pyplot as plt
import os


def save_evaluation_results(arima_metrics, sarima_metrics, arimax_metrics, filepath="results/evaluation_metrics.txt"):
    """
    Saves the evaluation results to a text file.

    :param arima_metrics: dict, RMSE and MAE for the ARIMA model.
    :param sarima_metrics: dict, RMSE and MAE for the SARIMA model.
    :param arimax_metrics: dict, RMSE and MAE for the ARIMAX model.
    :param filepath: str, file path where results will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        f.write("Model Evaluation Metrics\n")
        f.write("========================\n\n")
        f.write(
            f"ARIMA Model:\nRMSE: {arima_metrics['RMSE']:.4f}\nMAE: {arima_metrics['MAE']:.4f}\n\n")
        f.write(
            f"SARIMA Model:\nRMSE: {sarima_metrics['RMSE']:.4f}\nMAE: {sarima_metrics['MAE']:.4f}\n\n")
        f.write(
            f"ARIMAX Model:\nRMSE: {arimax_metrics['RMSE']:.4f}\nMAE: {arimax_metrics['MAE']:.4f}\n")

    print(f"\nEvaluation metrics saved to: {filepath}")


def plot_predictions(y_test, y_pred, model_name, results_folder="results"):
    """
    Plots test vs predicted values and saves the plot.

    :param y_test: pandas Series, actual test values.
    :param y_pred: pandas DataFrame, predicted values.
    :param model_name: str, name of the model (e.g., "ARIMA").
    :param results_folder: str, directory to save the plot.
    """
    # Ensure the results directory exists
    os.makedirs(results_folder, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label="Test", color='black')
    plt.plot(y_pred, label="Predictions", color='red', linestyle="dashed")

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Interest Rate")
    plt.title(f"{model_name} Prediction")
    plt.legend()

    # Save plot
    plot_filepath = os.path.join(
        results_folder, f"{model_name}_prediction.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Plot saved to: {plot_filepath}")
