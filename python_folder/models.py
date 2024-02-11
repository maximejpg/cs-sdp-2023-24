import pickle
from abc import abstractmethod
from gurobipy import Model, GRB

import keras
from keras import backend
from keras.layers import Activation, Add, Dense, Input, Lambda, Dropout, Subtract
from keras.models import Model, Sequential
from keras.utils import plot_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import os


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        model = None
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Add your own implementation here
        # For example, you can use the predict method of your model to calculate the utility
        utility = self.model.predict(X)

        return utility

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack(
            [np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1
        )


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.L = n_pieces
        self.K = n_clusters
        self.seed = 123
        # self.model = self.instantiate()
        self.model = Model("UTA model")
        self.criterions = 4
        self.breakpoints = []
        self.breakpoint_utils = {}

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        weights_1 = np.full(self.criterions, 1 / self.criterions)
        weights_2 = np.full(self.criterions, 1 / self.criterions)
        self.weights = [weights_1, weights_2]
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        self.breakpoints = [(1 / (self.L)) * i for i in range(self.L + 1)]

        P = len(X)
        I = {}
        for p in range(P):
            for k in range(self.K):
                I[p, k] = self.model.addVar(vtype=GRB.BINARY, name=f"I_{p}_{k}")
        M = 10
        e = 10**-3
        sigma = {}
        for p in range(P):
            for k in range(self.K):
                sigma[p, k] = self.model.addVar(
                    lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"sigma_{p}_{k}"
                )

        somme = 0
        # Variables for utility at each breakpoint
        breakpoint_utils = {}
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L + 1):
                    breakpoint_utils[k, i, b] = self.model.addVar(
                        lb=0,
                        ub=GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"breakpoint_utils_{k}_{i}_{b}",
                    )

        # Constraints for linear segments
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L):
                    self.model.addConstr(
                        (breakpoint_utils[k, i, b + 1] - breakpoint_utils[k, i, b]) >= 0
                    )

        # Function to calculate utility
        def calculate_utility(k, features):
            utility = 0
            for i, feature in enumerate(features):
                for b in range(self.L):
                    if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                        utility += breakpoint_utils[k, i, b] + (
                            (breakpoint_utils[k, i, b + 1] - breakpoint_utils[k, i, b])
                            / (self.breakpoints[b + 1] - self.breakpoints[b])
                        ) * (feature - self.breakpoints[b])
                        break
            return utility

        # Utility difference constraint
        for p in range(P):
            self.model.addConstr(sum(I[p, k] for k in range(self.K)) >= 1)
            for k in range(self.K):
                self.model.addConstr(
                    M * (1 - I[p, k])
                    + calculate_utility(k, X[p])
                    - calculate_utility(k, Y[p])
                    - e
                    + sigma[p, k]
                    >= 0
                )
        for p in range(P):
            for k in range(self.K):
                somme += sigma[p, k]
        self.model.setObjective(somme, GRB.MINIMIZE)
        self.breakpoint_utils = breakpoint_utils
        self.model.optimize()

        return

    def predict_utility(self, X):
        utilities = np.zeros(
            (len(X), self.K)
        )  # Tableau 2D: lignes pour les échantillons, colonnes pour les clusters
        for p in range(len(X)):
            for k in range(self.K):
                utility = 0
                for i, feature in enumerate(X[p]):
                    for b in range(self.L):
                        if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                            # Calculer l'utilité pour chaque cluster séparément
                            utility += self.breakpoint_utils[k, i, b].X + (
                                (
                                    self.breakpoint_utils[k, i, b + 1].X
                                    - self.breakpoint_utils[k, i, b].X
                                )
                                / (self.breakpoints[b + 1] - self.breakpoints[b])
                            ) * (feature - self.breakpoints[b])
                            break
                utilities[p, k] = (
                    utility  # Stocker l'utilité de l'échantillon 'p' pour le cluster 'k'
                )
        return utilities

        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def create_meta_network(self) -> Model:
        """
        Creates a meta network model that takes two inputs and predicts the probability of their relationship.

        Returns:
            Model: The meta network model.
        """
        # Create a new base network instance for each meta model
        seq = Sequential()
        seq.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM,), activation="relu"))
        seq.add(Dropout(0.1))
        seq.add(Dense(64, activation="relu"))
        seq.add(Dropout(0.1))
        seq.add(Dense(32, activation="relu"))
        seq.add(Dense(1, activation="sigmoid"))  # Apply sigmoid activation function

        input_a = Input(shape=(self.INPUT_DIM,))
        input_b = Input(shape=(self.INPUT_DIM,))

        rel_score = seq(input_a)
        irr_score = seq(input_b)

        # subtract scores
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        model = Model(inputs=[input_a, input_b], outputs=prob)
        model.compile(optimizer="adam", loss="binary_crossentropy")

        return model

    def __init__(self):
        """Initialization of the Heuristic Model."""
        self.seed = 123
        self.INPUT_DIM = 10
        self.K = 3
        self.history = []
        self.models = self.instantiate()

    def instantiate(self) -> list[Model]:
        """Instantiation of the MIP Variables"""
        # Initialize the clustering model
        self.kmeans = KMeans(n_clusters=self.K, random_state=self.seed)

        # Initialize the siamese networks
        models = []
        for _ in range(0, self.K):  # Create K models
            models.append(self.create_meta_network())
        return models

    def fit(self, X, Y):
        """Estimation of the parameters for each cluster model.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements.
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements.
        """
        # Différence X - Y pour déterminer le clustering
        diff = X - Y
        self.kmeans.fit(diff)

        # Sépare les données en fonction des étiquettes de cluster
        cluster_labels = self.kmeans.labels_

        for k in range(self.K):
            # Filtre les données pour le cluster k
            cluster_indices = np.where(cluster_labels == k)[0]
            X_k = X[cluster_indices]
            Y_k = Y[cluster_indices]

            if len(X_k) == 0:  # Vérifie s'il y a des données pour le cluster
                continue  # Passe au cluster suivant s'il n'y a pas de données

            # Configure EarlyStopping
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="auto"
            )

            # Prépare les labels pour l'entraînement: tous les X sont préférés à Y
            y_compare = np.ones(len(X_k))

            # Entraîne le modèle pour le cluster k
            history = self.models[k].fit(
                [X_k, Y_k],
                y_compare,
                validation_split=0.2,  # Utilise une fraction des données pour la validation
                epochs=50,
                batch_size=10,
                verbose=1,
                callbacks=[es],
            )

            self.history.append(history)

    def plot_history(self):
        """Plot the history of the model training."""
        for history in self.history:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper left")
            plt.show()

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.

        # Create a dummy Y to predict the utility (because needs 2 inputs)
        dummy_Y = np.zeros_like(X)
        # Predict the utility (preference score) for X using each model
        utility_scores = []
        for model in self.models:
            # The model expects two inputs; X and a dummy Y
            scores = model.predict([X, dummy_Y])
            utility_scores.append(scores)

        # # Convert list of scores to a numpy array for easy manipulation
        # utility_scores = np.array(utility_scores).squeeze(
        #     axis=0
        # )  # Adjust dimensions if necessary

        # return utility_scores

        # Correctly convert list of scores to a numpy array for easy manipulation
        utility_scores = np.array(utility_scores)

        # After conversion, we can now check the shape and dimensions
        # Ensure the shape of utility_scores is correct for applying np.argmax
        if utility_scores.ndim == 2 and utility_scores.shape[0] != len(X):
            utility_scores = utility_scores.transpose()

        # Find the index of the best utility score for each sample
        best_indices = np.argmax(utility_scores, axis=0)

        # Since utility_scores might have models as rows and samples as columns, we use best_indices to select best scores
        # Ensure to select best scores correctly based on the shape of utility_scores
        if utility_scores.shape[0] == len(X):
            # If utility_scores is shaped (samples, models), use best_indices directly
            best_utility = np.array(
                [utility_scores[i, best_indices[i]] for i in range(len(best_indices))]
            )
        else:
            # If utility_scores is shaped differently, adjust selection logic accordingly
            best_utility = utility_scores[best_indices, np.arange(len(best_indices))]

        return best_utility

    def predict_preference(self, X, Y) -> np.ndarray:
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        utilities = []
        for model in self.models:
            utilities.append(model.predict([X, Y]))

        return np.array(utilities)

        # # get the best preference
        # best_preference = np.argmax(utilities, axis=0)
        # return best_preference
        # # return np.array(utilities)

    def test_model(self, X, Y, Z):
        """Compare the preferences between X and Y
        Find the cluster that explain the best the preferences and compare the preferences with Z

        Args:
            X (np.ndarray): (n_samples, n_features) list of features of elements to compare with Y elements of same index
            Y (np.ndarray): (n_samples, n_features) list of features of elements to compare with X elements of same index
            Z (np.ndarray): (n_samples) list of preferences for each pair of elements
        """
        # Predict the preferences for each cluster
        preferences = self.predict_preference(X, Y)
        # Find the cluster with the highest utility
        best_cluster = np.argmax(preferences, axis=1)
        # Compare the preferences with Z
        accuracy = np.mean(best_cluster == Z)
        return accuracy

    def save_model_weights(self, directory_path):
        """Sauvegarde les poids des modèles dans le répertoire spécifié.

        Parameters:
        -----------
        directory_path: str
            Le chemin du répertoire où sauvegarder les poids des modèles.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        for idx, model in enumerate(self.models):
            model_path = os.path.join(directory_path, f"model_{idx}.h5")
            model.save_weights(model_path)
        print(f"Modèles sauvegardés dans {directory_path}")

    def load_model_weights(self, directory_path):
        """Charge les poids des modèles depuis le répertoire spécifié.

        Parameters:
        -----------
        directory_path: str
            Le chemin du répertoire d'où charger les poids des modèles.
        """
        for idx, model in enumerate(self.models):
            model_path = os.path.join(directory_path, f"model_{idx}.h5")
            if os.path.exists(model_path):
                model.load_weights(model_path)
                print(f"Poids chargés depuis {model_path}")
            else:
                print(f"Le fichier {model_path} n'existe pas, chargement impossible.")


# x - y marche si on a des utilité linaires et marche encore mieux du fait que les données sont monotonnes
# (et c'est le cas car nos données sont préférentielles)
