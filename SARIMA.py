import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools



df = pd.read_csv("dataset/bdd_prepared/BSS002EDYK.csv")

df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

df = df[df["Date"].dt.year >= 2010]
df.set_index("Date", inplace=True)
df_weekly = df.resample("W-MON").mean()

y = df_weekly["niveau_nappe_eau"].astype(float)

y_train = y['2014':'2020']
y_test = y['2021':]

period = 52  # Période hebdomadaire pour des données journalières


decomposition = seasonal_decompose(y, model='mult', period=period)
decomposition.plot()


fig, axes = plt.subplots(2, 1, figsize=(12, 6))

y_seasonal_diff = y_train.diff(period).dropna()
plot_acf(y_seasonal_diff, lags=period, ax=axes[0])
plot_pacf(y_seasonal_diff, lags=period, ax=axes[1])

plt.show()


def Calcule_hyperparamètres():
    p = d = q = range(0, 2)
    P = Q = range(0, 2)
    D = [1]

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, [period]))

    best_aic = np.inf
    best_param = None
    best_seasonal_param = None

    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                model = SARIMAX(
                    y_train,
                    order=param,
                    seasonal_order=seasonal_param,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_param = param
                    best_seasonal_param = seasonal_param
            except:
                continue

    return best_param, best_seasonal_param

def ExécuterSARIMA(order, seasonal_order):
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model


best_param, best_seasonal_param = (1, 1, 2), (0, 1, 1, 52)
# best_param, best_seasonal_param = Calcule_hyperparamètres()
print("Meilleurs hyperparamètres SARIMA :", best_param, best_seasonal_param)

model = ExécuterSARIMA(best_param, best_seasonal_param)

result = model.fit()
print(result.summary())

# test the model
y_pred_test = result.predict(start=len(y_train), end=len(y_train) + len(y_test)-1)
y_pred_test.index = y_test.index
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred_test, label='Predicted', color='red')
plt.legend()
plt.show()


plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()

#metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')