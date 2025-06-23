import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from sklearn.mixture import GaussianMixture
from umap import UMAP

class GMMLabelCorrector:
    def __init__(self, embeddings, n_components=2, covariance_type='full'):
        self.reducer = UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
        self.gmm.fit(embeddings)

    def threshold_predict(self, embeddings, threshold=0.3):
        """
        Predict labels based on GMM probabilities and a threshold.
        If the probability of the other class is above the threshold, assign it to that class.
        """
        probabilities = self.gmm.predict_proba(embeddings)
        # Flip to the less likely class if model is uncertain
        predictions = [np.argmin(x) if min(x) >= threshold else np.argmax(x) for x in probabilities]
        return predictions