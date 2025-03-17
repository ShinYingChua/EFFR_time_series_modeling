# Model Types
MODEL_TYPES = ["ARIMA", "SARIMA", "ARIMAX"]

# ARIMA Parameters (p, d, q)
ARIMA_PARAMS = {
    "order": (1, 0, 1)
}

ARIMAX_PARAMS = {
    "order": (1, 0, 1),  # ARIMAX (p, d, q) same as ARIMA
}

# SARIMAX Parameters (p, d, q) + Seasonal (P, D, Q, s)
SARIMA_PARAMS = {
    "order": (2, 0, 3),
    "seasonal_order": (0, 1, 0, 12)
}

# Train-Test Split Ratio
TRAIN_RATIO = 0.8

# Target Variable
TARGET_COLUMN = "Interest_Rate"

# Exogenous Variables
EXOGENOUS_COLUMNS = ["Inflation_Rate", "Unemployment_Rate"]
