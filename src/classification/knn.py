from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k: int = 5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)