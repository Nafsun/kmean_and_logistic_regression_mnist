from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

X_digits, y_digits = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

pipeline = Pipeline([
    ("kmeans", KMeans()),
    ("log_reg", LogisticRegression(C=0.1, solver="liblinear")),
])

param_grid = dict(kmeans__n_clusters=range(50, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

print(grid_clf.best_params_) # {'kmeans__n_clusters': 97}

print(grid_clf.score(X_test, y_test)) # 0.9911111111111112