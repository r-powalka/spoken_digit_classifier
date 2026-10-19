from .gmm import GMM


class GMMClassifier:
    """
    Spoken-digit classifier using one GMM per digit.

    Parameters
    ----------
    C : int, default=5
        Number of Gaussian components in each GMM.

    seed : int or None, default=None
        Random seed used for GMM initialization.
    """

    def __init__(self, C=5, seed=None):
        self.C = C
        self.seed = seed

        self.models = {}
        self.is_fit = False

    def fit(
        self,
        X_train,
        max_iter=100,
        tol=1e-3,
        verbose=False,
    ):
        """
        Train one GMM for each digit.

        Parameters
        ----------
        X_train : dict
            Dictionary mapping digit labels to MFCC feature
            matrices.

            Example:
                {
                    0: X_0,
                    1: X_1,
                    ...
                    9: X_9
                }

        max_iter : int, default=100
            Maximum number of EM iterations for each GMM.

        tol : float, default=1e-3
            Relative convergence tolerance.

        verbose : bool, default=False
            Whether to print GMM training progress.

        Returns
        -------
        self
            Fitted classifier.
        """

        self.models = {}

        for label, X in sorted(X_train.items()):

            print(f"Training digit {label}...")

            model = GMM(
                C=self.C,
                seed=self.seed,
            )

            model.fit(
                X,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )

            self.models[label] = model

        self.is_fit = True

        return self

    def score(self, X):
        """
        Compute the score of an utterance under every
        digit GMM.

        Parameters
        ----------
        X : ndarray
            MFCC features for one utterance.

        Returns
        -------
        dict
            Mapping from digit label to mean log-likelihood.
        """

        if not self.is_fit:
            raise RuntimeError(
                "Classifier must be fitted before scoring."
            )

        return {
            label: model.score(X)
            for label, model in self.models.items()
        }

    def predict(self, X):
        """
        Predict the digit corresponding to an utterance.

        Parameters
        ----------
        X : ndarray
            MFCC features for one utterance.

        Returns
        -------
        int
            Predicted digit.
        """

        scores = self.score(X)

        return max(
            scores,
            key=scores.get,
        )