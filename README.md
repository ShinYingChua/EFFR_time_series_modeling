# Effective Federal Funds Rate (EFFR) Modelling Using ARIMA/ SARIMA/ ARIMAX
Interest rates play a crucial role in financial markets, influencing borrowing costs, investments, and economic growth. The Effective Federal Funds Rate (EFFR) is a key benchmark that reflects the cost of overnight lending between banks and serves as a foundation for setting broader loan and mortgage interest rates.

This project focuses on evaluating the performance of different time series models in capturing historical EFFR trends. We explore three statistical modeling approaches:

- ARIMA (AutoRegressive Integrated Moving Average): Captures non-seasonal trends in interest rates.
- SARIMA (Seasonal ARIMA): Accounts for seasonal patterns in rate fluctuations.
- ARIMAX (ARIMA with Exogenous Variables): Incorporates macroeconomic factors like inflation and unemployment rates to enhance prediction accuracy.

By analyzing historical data from 2007 to 2017, this project provides insights into the strengths and limitations of different time series forecasting models in capturing EFFR behavior. 

## Dataset
This dataset includes data on the economic conditions in the United States on a monthly basis since 1954. The federal funds rate is the interest rate at which depository institutions trade federal funds (balances held at Federal Reserve Banks) with each other overnight.

- Source: [Federal Reserve Economic Data (FRED)](https://www.kaggle.com/datasets/federalreserve/interest-rates?resource=download)
- Ful timeframe: 1954 - 2017
- Granularity: Monthly data
- Key Features:
    - Effective Federal Funds Rate (Main column used for modeling)
    - Unemployment Rate
    - Inflation Rate

The timeframe considered for this project is 2007 - 2017.
- This is resonable becasue the 10-year period covers multiple rate hikes and cuts, ensuring that the model learns from both rising and falling interest rate environments. It captures: <br/>
      - The Global Financial Crisis (2008-2009) and near-zero interest rates. <br/>
      - 2015–2017 Federal Reserve rate hikes → Post-crisis normalization.

- Box & Jenkins (1970) recommend at least 50 data points for reliable ARIMA modeling. With 120 observations (10 years × 12 months), this condition is satified. SARIMA models require multiple seasonal cycles. With 10 full seasonal cycles (seasonality=12 months), the model can accurately capture seasonal effects.

## Overview of folder structure
    .
    ├── .github   
    ├── data
         └── interest_rates.csv 
    ├── src
         └── config.py
         └── dataloader.py
         └── models.py
         └── main.py
         └── preprocessing.py
         └── utils.py
    ├── .gitignore
    ├── README.md                   
    ├── eda.ipynb 
    ├── requirements.txt                                    
                 
## Instructions for executing pipeline:

1. Customise requirements.txt for development/staging environment:
   - Amend `requirements.txt` by commenting out the section.
2. Execute pipeline:
   - GitHub Actions workflow: Whenever user push changes to the repository (any branch), the workflow will run. 
   - Manually using workflow_dispatch: manually trigger the workflow via the GitHub Actions UI.
   - or excute `./run.sh` in command line locally.

## Key findings of EDA
1) Data prepocessing 
- Convert to datetime format to ensure proper time series handling.
- Standardize column names to maintain clarity and consistency across datasets.
- Filter data for the last 10 years to focus on the most relevant trends and pattern.
- Handle missing values by dropping
  - Reason for NA values: The missing values occur due to duplicate entries within the same year and month but on different days.
  - Since we are working with monthly data, removing these duplicates does not affect the overall trend.
  - Interest rates are policy-driven and may experience sudden changes, making imputation unreliable.
  <img width="987" alt="image" src="https://github.com/user-attachments/assets/fd375489-7b83-4cfc-95e0-7df3b7d0ff7e" />

2) Insights
- Rolling Mean & Standard Deviation
    - Some periods (e.g., 2009–2015) had stable rates, making them stationary.
    - Other periods (e.g., 2007–2009 and 2015–2017) had rapid changes, making them non-stationary.
  <img width="987" alt="image" src="https://github.com/user-attachments/assets/2311e85e-96cb-4423-99b3-b400191bf6f4" />

- Global Stationarity Check
    - While ARIMA models can deal with non-stationarity up to a point, they cannot effectively account for time-varying variance. In other words, for an ARIMA model to really work, the data has to be stationary. 
    - Augmented Dickey-Fuller (ADF) test is perifrmed and the series is found to be stationary (p-value <=0.05)
    - While some periods (e.g., 2007–2009, 2015–2017) exhibit volatility or structural changes, the dataset as a whole remains statistically stable. Therefore, differencing is not required.
      
 - Autocorrelation
    - ACF: Plot shows a gradual decline instead of a sharp cutoff (q = 0 or 1).
      <img width="987" alt="image" src="https://github.com/user-attachments/assets/d7537b38-f4b3-4128-baae-f4f6b4d22a2f" />

    - PACF: Plot cuts off sharply after lag 1 (p=1)
      <img width="987" alt="image" src="https://github.com/user-attachments/assets/95902d2b-08eb-4b8f-ba55-b5cff759c3c4" />

    - ADF test confirms stationarity (p-value < 0.05), set d=0
    - Try ARIMA parameters (1, 0, 1) 
         
## Model Choice
1) ARIMA (Baseline model)
- ARIMA is defined by three parameters:
    \[
    ARIMA(p, d, q)
    \]
    - **\(p\) (AutoRegressive term)** – Number of past observations (lags) used for prediction.
    - **\(d\) (Differencing term)** – Number of times the data needs differencing to become stationary.
    - **\(q\) (Moving Average term)** – Number of past forecast errors included in the model.
  
2) SARIMA
- SARIMA is considered because it enhances the modeling of EFFR by capturing both trend and seasonality in interest rate movements. Interest rates often exhibit seasonal or cyclical behavior due to economic cycles, monetary policy changes, and market expectations.
- Based on the seasonality plot ,the peaks and troughs suggest a recurring pattern over time, which justifies using SARIMA 
  <img width="987" alt="image" src="https://github.com/user-attachments/assets/a6d85162-0279-4f11-ac26-6b331adebaf6" />
-  Enforce d=0 (no trend differencing) and D=1 (one seasonal difference) while optimizing the remaining parameters using Optuna to minimize RMSE.
-  Order(p, d, q) = (2, 0, 3) and Seasonal Order(P, D, Q, s) = (0, 1, 0 , 12)

3) ARIMAX
- The Federal Reserve adjusts interest rates based on inflation, unemployment, and economic growth. ARIMAX allows us to include these exogenous variables as predictors, improving model performance.
- Exogenous Variables: 
    - **Inflation Rate**  – Higher inflation often leads the Federal Reserve to raise interest rates to maintain price stability.  
    - **Unemployment Rate** – The Fed balances inflation and employment; higher unemployment may lead to rate cuts to stimulate economic activity.  
    - **GDP Growth Rate (Excluded)** – Although GDP growth is a relevant factor, it has **high missingness in the dataset**, which could introduce bias and reduce model reliability.
- ARIMA only considers past values of EFFR, but ARIMAX combines past values with external influences. This makes ARIMAX more realistic and responsive to economic policy shifts.

## Results

1) Model Accuracy
RMSE and MAE is used to evaluate the models. MAE gives a straightforward idea of average model error. RMSE emphasizes larger errors, ensuring that the model doesn't make drastic mistakes.

| Model                                 | Parameters                                                       | RMSE          | MAE     |                                                                      
|--------------------                   |---------------------------------------------                     | -----         | ----    |
| ARIMA (baseline Model)                | p, d, q = (1,0,1)                                                | 0.1542        | 0.1148  | 
| SARIMA                                | p, d, q = (2,0,3), P, D, Q = (0, 1, 0, 12)                       | 0.2090        | 0.1552  | 
| ARIMAX                                | p, d, q = (1,0,1) + Exogenous Variables                          | 0.1095        | 0.0809  | 

ARIMAX is the best model, as it has the lowest RMSE (0.1095) and MAE (0.0809). This suggests that including exogenous variables (e.g., inflation, unemployment) improves prediction accuracy.
SARIMA performs the worst, likely because the seasonal component does not provide additional explanatory power in this dataset.

Interest rates are largely policy-driven, meaning rate changes depend more on economic conditions than strict seasonality. Decomposition plot shows some seasonality, but ACF/PACF do not confirm strong seasonal lags (e.g., 12, 24, etc.).
The lack of repeating seasonal spikes suggests that seasonality is weak or inconsistent in the dataset.Since SARIMA assumes a consistent seasonal pattern, it struggles to capture the dynamics of interest rates effectively.

2) Prediction VS Testing
- ARIMA model shows a linear trend in predictions 
<img width="987" alt="image" src="https://github.com/user-attachments/assets/2ee76729-66ee-4346-ba9e-5ae2a0b69eb9" />

- SARIMA predictions remain almost flat, indicating that the model is unable to capture trend changes or significant variations in the data.
<img width="987" alt="image" src="https://github.com/user-attachments/assets/12af9bda-0096-497e-a50c-890c4a63e74d" />

- ARIMAX mode captures the general trend and rate fluctuations more accurately than ARIMA or SARIMA.
<img width="987" alt="image" src="https://github.com/user-attachments/assets/c8e56e17-8fd1-45e2-8cc5-2e0985585bc2" />

## Reference
1) https://www.datacamp.com/tutorial/arima
2) https://www.kaggle.com/code/vipin20/arima-sarimax-exponential-smoothing-using-optuna
