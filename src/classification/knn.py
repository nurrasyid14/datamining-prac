from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, data, target, k: int = 5):
        self.X_train = data.copy()
        self.y_train = target.copy()
        self.k = k

        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)