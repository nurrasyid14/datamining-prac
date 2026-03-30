from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, data, target, k: int = 5, test_size: float = 0.2, random_state: int = 42):
        self.data = data.copy()
        self.target = target
        self.k = k
        self.test_size = test_size
        self.random_state = random_state

        self.model = KNeighborsClassifier(n_neighbors=self.k)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split(self):
        X = self.data.drop(columns=[self.target])
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        if self.X_train is None:
            self.split()

        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self):
        if self.X_test is None:
            raise ValueError("Data not split. Call split() first.")

        return self.model.predict(self.X_test)

    def predict_proba(self):
        if self.X_test is None:
            raise ValueError("Data not split. Call split() first.")

        return self.model.predict_proba(self.X_test)