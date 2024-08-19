import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import nest_asyncio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

# Permitir el anidamiento de bucles de eventos
nest_asyncio.apply()

async def download(url, filename):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, "wb") as f:
                    f.write(await response.read())


path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
file_name="laptops.csv"
width = 12
height = 10

async def main():
    await download(path, file_name)

# Ejecutar la función asincrónica
asyncio.run(main())


df = pd.read_csv(file_name, header=0)

# Linear Regression
lm = LinearRegression()
X = df[['CPU_frequency']]
Y = df['Price']

# Fit Model
lm.fit(X, Y)

# Output a Predict
Yhat = lm.predict(X)
print("Output a Predict of lm model")
print(Yhat[0:5])



#Distribution Plot to visualize Multiple Line Regression Model Efficiency
plt.figure(figsize=(width, height))
# Gráfico de densidad para valores reales
sns.kdeplot(df['Price'], color="r", label="Actual Value")

# Gráfico de densidad para valores ajustados
sns.kdeplot(Yhat, color="b", label="Predicted Value")

plt.title('Actual vs Fitted Values for Price')
plt.legend()
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Laptops')

plt.show()


# R2 and MSE of SLR
print('The R-square is: ', lm.score(X, Y))
mse = mean_squared_error(df['Price'], Yhat)
print('The mean square error of price and predicted value for a Single Line Regresion is: ', mse)

#Multiple Regression
Z=df[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']]
lm2 = LinearRegression()

#Fit the model
lm2.fit(Z, df['Price'])
Y_hat = lm2.predict(Z)
print("Predicition using the Multiple Line Regression")
print(Y_hat[0:14])

#Distribution Plot to visualize Multiple Line Regression Model Efficiency
plt.figure(figsize=(width, height))
# Gráfico de densidad para valores reales
sns.kdeplot(df['Price'], color="r", label="Actual Value")

# Gráfico de densidad para valores ajustados
sns.kdeplot(Y_hat, color="b", label="Predicted Value")

plt.title('Actual vs Fitted Values for Price')
plt.legend()
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Laptops')

plt.show()

#R2 and MSE of MLR
print('The R-square  for a Multiple Line Regresion is: ', lm2.score(Z, df['Price']))
mse = mean_squared_error(df['Price'], Y_hat)
print('The mean square error of price and predicted value for a Multiple Line Regresion is: ', mse)
#The values of R2 and MSE for Multiple Line Regresion are better than the ones for Single Line Regression, R2_MLR bigger than R2_SLR and MSE_MLR less than MSE_SLR

#Polynomial Regression
x = df['CPU_frequency']
y = df['Price']
f1 = np.polyfit(x, y, 1)
p1 = np.poly1d(f1)

f2 = np.polyfit(x, y, 3)
p2 = np.poly1d(f2)

f3 = np.polyfit(x, y, 5)
p3 = np.poly1d(f3)

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')


# Call for function of degree 1
PlotPolly(p1, x, y, 'CPU_frequency')
plt.show()

# Call for function of degree 3
PlotPolly(p2, x, y, 'CPU_frequency')
plt.show()

# Call for function of degree 5
PlotPolly(p3, x, y, 'CPU_frequency')
plt.show()


#Pipelines
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
#First, we convert the data type Z to type float to avoid conversion warnings
# that may appear as a result of StandardScaler taking float inputs.
#Then, we can normalize the data, perform a transform and fit the model simultaneously.
Z = Z.astype(float)
pipe.fit(Z,y)
#Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
ypipe=pipe.predict(Z)
print("Prediction made by the pipeline method")
print(ypipe[0:4])

#R2 and MSE of Pipeline:
print("R2 of pipe Linear Regression is: ",pipe.score(Z,df["Price"]) )
print("MSE of pipe Linear Regression is: ",mean_squared_error(df['Price'], ypipe) )



#R2 and MSE for Poly Regression
r_squared_1 = r2_score(y, p1(x))
r_squared_2 = r2_score(y, p2(x))
r_squared_3 = r2_score(y, p3(x))
print('The R-square value for a Polynomial Regresion Order 1 is: ', r_squared_1)
print("The mean square error  of price and predicted value for a Polynomial Regresion Order 1 is: ",mean_squared_error(df['Price'], p1(x)))

print('The R-square value for a Polynomial Regresion Order 3 is: ', r_squared_2)
print("The mean square error  of price and predicted value for a Polynomial Regresion Order 3 is: ",mean_squared_error(df['Price'], p2(x)))

print('The R-square value for a Polynomial Regresion Order 5 is: ', r_squared_3)
print("The mean square error  of price and predicted value for a Polynomial Regresion Order 5 is: ",mean_squared_error(df['Price'], p3(x)))

print("The Best Model is Polynomial Regression Order 5 since it has the highest R2 and the lowest MSE")





def DistributionPlot(predicted_value):
    plt.figure(figsize=(width, height))
    # Gráfico de densidad para valores reales
    sns.kdeplot(df['Price'], color="r", label="Actual Value")

    # Gráfico de densidad para valores ajustados
    sns.kdeplot(predicted_value, color="b", label="Predicted Value")
    plt.show()

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(Z, df['Price'])
dt_predictions = dt.predict(Z)
print('Decision Tree R2:', dt.score(Z, df['Price']))
print('Decision Tree MSE:', mean_squared_error(df['Price'], dt_predictions))

# Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(Z, df['Price'])
rf_predictions = rf.predict(Z)
print('Random Forest R2:', rf.score(Z, df['Price']))
print('Random Forest MSE:', mean_squared_error(df['Price'], rf_predictions))

# Gradient Boosting Regression
gb = GradientBoostingRegressor()
gb.fit(Z, df['Price'])
gb_predictions = gb.predict(Z)
print('Gradient Boosting R2:', gb.score(Z, df['Price']))
print('Gradient Boosting MSE:', mean_squared_error(df['Price'], gb_predictions))

# Support Vector Regression (SVR)
svr = SVR(kernel='rbf')
svr.fit(Z, df['Price'])
svr_predictions = svr.predict(Z)
print('SVR R2:', svr.score(Z, df['Price']))
print('SVR MSE:', mean_squared_error(df['Price'], svr_predictions))

# Neural Network
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
mlp.fit(Z, df['Price'])
mlp_predictions = mlp.predict(Z)
print('Neural Network R2:', mlp.score(Z, df['Price']))
print('Neural Network MSE:', mean_squared_error(df['Price'], mlp_predictions))



#Decision Tree Distribution Plot:
DistributionPlot(rf_predictions)

#Random Forest Distribution Plot:
DistributionPlot(rf_predictions)

#Gradient Boosting Distribution Plot:
DistributionPlot(gb_predictions)

#Support Vector Regression (SVR) Distribution Plot:
DistributionPlot(svr_predictions)

#Neural Neywork Distribution Plot:
DistributionPlot(mlp_predictions)


# Updated Polynomial Regression
polynomial_pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=5, include_bias=False)),
    ('linear_regression', LinearRegression())
])

polynomial_pipeline.fit(Z, df['Price'])
polynomial_predictions = polynomial_pipeline.predict(Z)
print('Polynomial Regression (Degree 5) R2:', polynomial_pipeline.score(Z, df['Price']))
print('Polynomial Regression (Degree 5) MSE:', mean_squared_error(df['Price'], polynomial_predictions))




models = {
    'Linear Regression': lm2,
    'Polynomial Regression (Degree 5)': polynomial_pipeline,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'SVR': svr,
    'Neural Network': mlp
}

for name, model in models.items():
    scores = cross_val_score(model, Z, df['Price'], cv=5, scoring='r2')
    print(f'{name} R2 (cross-validated): {scores.mean()}')

    predictions = model.predict(Z)
    mse = mean_squared_error(df['Price'], predictions)
    print(f'{name} MSE: {mse}')
