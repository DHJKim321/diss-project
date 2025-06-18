from sklearn.mixture import GaussianMixture

class GMMModel:
    def __init__(self, n_components=2, covariance_type='full', random_state=42):
        """
        Initialize the Gaussian Mixture Model.

        :param n_components: Number of mixture components.
        :param covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
        :param random_state: Random seed for reproducibility.
        """
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.fitted = False

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to the data.
        :param X: Input data, shape (n_samples, n_features).
        """
        self.model.fit(X)
        self.fitted = True

    def predict(self, X):
        """
        Predict the labels for the input data.
        :param X: Input data, shape (n_samples, n_features).
        :return: Predicted labels, shape (n_samples,).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)