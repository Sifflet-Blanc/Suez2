import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def get_model(attribute, print_r2=False, n_compo=32):
    X_train, X_test, y_train, y_test = train_test_split(
        attribute.drop(columns=["water_level_mean"]), attribute["water_level_mean"], test_size=0.2
    )
    
    # Reduction and center the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    V = np.array(scaler.scale_)
    E = np.array(scaler.mean_)

    # Apply PCA to reduce the number of component for the regression
    pca = PCA(n_components=n_compo)
    X_train_pca = pca.fit_transform(X_train_scaled)
    A = np.array(pca.components_)
    
    # Create and fit the regression model
    X2 = sm.add_constant(list(X_train_pca))
    model = sm.OLS(list(np.array(y_train)), X2)
    model = model.fit()

    # The function to be apply on one data and return the estimate water level
    def f(X):
        return A.dot((X - E) / V)
    
    def predict(X):
        npX = np.array(X)
        if len(npX.shape) == 1:
            return model.predict(sm.add_constant(f(npX)))
        elif len(npX.shape) == 2:
            return model.predict(sm.add_constant([f(x) for x in npX]))
        return None
    
    if print_r2:
        model.summary()

        y_pred_train = predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        y_pred = predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("MSE on trainning : "+str(mse_train))
        print("R2 on trainning : "+str(r2_train))
        print()
        print("MSE on test : "+str(mse))
        print("R2 on test : "+str(r2))

    return predict