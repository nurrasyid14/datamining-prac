from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ClusteringMetrics:
    @staticmethod
    def silhouette(X, labels):
        return silhouette_score(X, labels)

    @staticmethod
    def davies_bouldin(X, labels):
        return davies_bouldin_score(X, labels)

    @staticmethod
    def calinski_harabasz(X, labels):
        return calinski_harabasz_score(X, labels)