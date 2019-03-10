import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data
y = iris.target
# select features
X = X[y < 2, :2]
y = y[y < 2]

# plot
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue")
plt.show()

# split training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

# own function
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print("final score is :%s" % log_reg.score(X_test, y_test))
print("actual prob is :")
print(log_reg.predict_proba(X_test))
