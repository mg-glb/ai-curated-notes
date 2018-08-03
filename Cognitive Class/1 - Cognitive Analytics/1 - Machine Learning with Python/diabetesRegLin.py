from sklearn.datasets import load_diabetes 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

diabetes = load_diabetes()
diabetes_X = diabetes.data[:, None, 2]
LinReg = LinearRegression()

X_trainset, X_testset, y_trainset, y_testset = train_test_split(diabetes_X, diabetes.target, test_size=0.3, random_state=7)
LinReg.fit(X_trainset,y_trainset)

y_predict = LinReg.predict(X_testset)

print("Linear Regression MAE: %r" % metrics.mean_absolute_error(y_testset,y_predict))
print("Linear Regression MSE: %r" % metrics.mean_squared_error(y_testset,y_predict))
print("Linear Regression RMSE: %r" % (metrics.mean_squared_error(y_testset,y_predict) ** (0.5)))