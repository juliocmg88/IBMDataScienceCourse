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

# Permitir el anidamiento de bucles de eventos
nest_asyncio.apply()

async def download(url, filename):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, "wb") as f:
                    f.write(await response.read())

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
file_name = "usedcars.csv"

width = 12
height = 10

async def main():
    await download(file_path, file_name)

# Ejecutar la función asincrónica
asyncio.run(main())

# Leer el archivo CSV
df = pd.read_csv(file_name)
print(df.info())

# Asegurarse de que las columnas no tengan valores NaN y sean numéricas
#df = df[['highway-mpg', 'price']].dropna()
#df['highway-mpg'] = pd.to_numeric(df['highway-mpg'], errors='coerce')
#df['price'] = pd.to_numeric(df['price'], errors='coerce')
#df = df.dropna()

# Linear Regression
lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']

# Fit Model
lm.fit(X, Y)

# Output a Predict
Yhat = lm.predict(X)
print("Output a Predict of lm model")
print(Yhat[0:5])

coef=lm.coef_
intercept=lm.intercept_
print(f"the lm model's value of the intercept is:  {intercept} and the slope is: {coef}")

#Linear Regression By Engine Size:
lm1 = LinearRegression()

#Fit the model lm1
X1 = df[["engine-size"]]
Y1 = df[["price"]]
lm1.fit(X1,Y1)

coef1 = lm1.coef_
intercept1 = lm1.intercept_
print(f"the lm1 model's value of the intercept is:  {intercept1} and the slope is: {coef1}")

yhatl1 = intercept1+coef1*X1
yhatl1.rename(columns={'engine-size': 'predicted_value'}, inplace=True)

# Agregar la columna de precio real
yhatl1['price'] = Y1.values

# Iterar sobre los valores de yhatl1
for index, row in yhatl1.iterrows():
    print(f"Predicted Value: {row['predicted_value']}, Actual Price: {row['price']}")

#Multiple Regression
#Z are independent variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm2 = LinearRegression()

#Fit the model
lm2.fit(Z, df['price'])
coef2 = lm2.coef_
intercept2 = lm2.intercept_
print(f"the lm2 model's value of the intercept is:  {intercept2} and the slope is: {coef2}")


#Predicition using the Multiple Line Regression
Y_hat = lm2.predict(Z)
print("Predicition using the Multiple Line Regression")
print(Y_hat[0:14])

#Distribution Plot to visualize Multiple Line Regression Model Efficiency
plt.figure(figsize=(width, height))
# Gráfico de densidad para valores reales
sns.kdeplot(df['price'], color="r", label="Actual Value")

# Gráfico de densidad para valores ajustados
sns.kdeplot(Y_hat, color="b", label="Fitted Values")

plt.title('Actual vs Fitted Values for Price')
plt.legend()
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
#We can see that the fitted values are reasonably close to the actual values
# since the two distributions overlap a bit. However, there is definitely some room for improvement.



#Regression Plot of Dataframe Using highway-mpg as independent variable

plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.title("Regression Plot of Highway MPG vs. Price")
plt.ylim(0,)
plt.show()

#Regression Plot of Dataframe Using peak-rpm as independent variable
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.title("Regression Plot of Peak RPM vs. Price")
plt.ylim(0,)
plt.show()


#Comparing the regression plot of "peak-rpm" and "highway-mpg",
# we see that the points for "highway-mpg" are much closer to the generated line and, on average, decrease.
# The points for "peak-rpm" have more spread around the predicted line and it is much harder to determine
# if the points are decreasing or increasing as the "peak-rpm" increases.

#Correlation comparison between peak-rpm and highway-mpg against price
correlation=df[["peak-rpm","highway-mpg","price"]].corr()
print(correlation)

#Residual Plot of Dataframe Using highway-mpg as independent variable
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.title("Residual Plot of Highway MPG vs. Price")
plt.show()

#What is this plot telling us?
# We can see from this residual plot that the residuals are not randomly spread around the x-axis,
# leading us to believe that maybe a non-linear model is more appropriate for this data.


#Define Function to plot a Polinomial Regression
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']
#Let's fit the polynomial using the function polyfit,
# then use the function poly1d to display the polynomial function.
# Here we use a polynomial of the 3rd order (cubic)
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

#Plot the function
PlotPolly(p, x, y, 'highway-mpg')
#We can already see from plotting that this polynomial model performs better than the linear model.
# This is because the generated polynomial function "hits" more of the data points.

#Create 11 order polynomial model with the variables x and y from above.
f11 = np.polyfit(x, y, 11)
p1 = np.poly1d(f11)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')

#We can perform a polynomial transform on multiple features.
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
print("In the original data, there are 201 samples and 4 features. ",Z.shape)
print("After the transformation, there are 201 samples and 15 features.",Z_pr.shape)

#Data Pipelines simplify the steps of processing the data.
# We use the module Pipeline to create a pipeline.
# We also use StandardScaler as a step in our pipeline.
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




#R2 for a Single Linear Regression
#R squared, also known as the coefficient of determination, is a measure to indicate how close the data is
# to the fitted regression line.
# The value of the R-squared is the percentage of variation of the response variable (y) that is explained by
# a linear model.
# Mean Squared Error (MSE)
# The Mean Squared Error measures the average of the squares of errors.
# That is, the difference between actual value (y) and the estimated value (ŷ).
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

#Let's calculate the MSE:
#We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
Yhat=lm.predict(X)
print('The output of the first four predicted value for a Single Line Regresion is: ', Yhat[0:4])

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value for a Single Line Regresion is: ', mse)


#R2 and MSE for a Multiple Line Regresion
# fit the model
lm2.fit(Z, df['price'])
# Find the R^2
print('The R-square for a Multiple Line Regresion is : ', lm2.score(Z, df['price']))

Y_predict_multifit = lm2.predict(Z)
print('The mean square error of price and predicted value using multifit for a Multiple Line Regresion is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))


#R2 and MSE for Polynomial Regresion
#Here we use the function r2_score from the module metrics as we are using a different function.
r_squared = r2_score(y, p(x))
print('The R-square value for a Polynomial Regresion is: ', r_squared)
print("The mean square error  of price and predicted value for a Polynomial Regresion is: ",mean_squared_error(df['price'], p(x)))

#Prediction and decision making
#In the previous section, we trained the model using the method fit.
# Now we will use the method predict to produce a prediction. Lets import pyplot for plotting;
# we will also be using some functions from numpy.
new_input = np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhati = lm.predict(new_input)
print("First Five Predicted Values for new_input are: ",yhati[0:5])

plt.plot(new_input, yhati)
plt.show()

"""
Decision Making: Determining a Good Model Fit
Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?
•	What is a good R-squared value?
When comparing models, the model with the higher R-squared value is a better fit for the data.
•	What is a good MSE?
When comparing models, the model with the smallest MSE value is a better fit for the data.
Let's take a look at the values for the different models.
Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.
•	R-squared: 0.49659118843391759
•	MSE: 3.16 x10^7
Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.
•	R-squared: 0.80896354913783497
•	MSE: 1.2 x10^7
Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.
•	R-squared: 0.6741946663906514
•	MSE: 2.05 x 10^7

Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)
Usually, the more variables you have, the better your model is at predicting, but this is not always true. Sometimes you may not have enough data, you may run into numerical problems, or many of the variables may not be useful and even act as noise. As a result, you should always check the MSE and R^2.
In order to compare the results of the MLR vs SLR models, we look at a combination of both the R-squared and MSE to make the best conclusion about the fit of the model.
•	MSE: The MSE of SLR is 3.16x10^7 while MLR has an MSE of 1.2 x10^7. The MSE of MLR is much smaller.
•	R-squared: In this case, we can also see that there is a big difference between the R-squared of the SLR and the R-squared of the MLR. The R-squared for the SLR (~0.497) is very small compared to the R-squared for the MLR (~0.809).
This R-squared in combination with the MSE show that MLR seems like the better model fit in this case compared to SLR.
Simple Linear Model (SLR) vs. Polynomial Fit
•	MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
•	R-squared: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.
Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting "price" with "highway-mpg" as a predictor variable.
Multiple Linear Regression (MLR) vs. Polynomial Fit
•	MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
•	R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.


Conclusion
Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset. This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.


"""
