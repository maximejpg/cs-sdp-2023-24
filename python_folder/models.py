import pickle
from abc import abstractmethod
from gurobipy import Model, GRB

import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
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
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


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
        self.L=n_pieces
        self.K=n_clusters
        self.seed = 123
        # self.model = self.instantiate()
        self.model = Model("UTA model")
        self.criterions = 4

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        weights_1 = np.full(self.criterions, 1/self.criterions)
        weights_2 = np.full(self.criterions, 1/self.criterions)
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

        self.breakpoints= [(1/(self.L))*i for i in range(self.L+1)]
       

        P = len(X)
<<<<<<< HEAD:python_folder/models.py
        I={}
        for p in range(P):
            for k in range(self.K):
                I[p, k] = self.model.addVar(vtype=GRB.BINARY, name=f"I_{p}_{k}")
        M = 10
        e = 10**-3
        sigma = {}
        for p in range(P):
            for k in range(self.K):
                sigma[p, k] = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"sigma_{p}_{k}")
=======
        # I = self.model.addVars(P, self.K, vtype=GRB.BINARY, name="I")
        I={}
        for p in range(P):
            for k in range(self.K):
                I[p,k] = self.model.addVar(vtype=GRB.BINARY)
        M = 10
        e = 10^-3
        sigma = {}
        for p in range(P):
            for k in range(self.K):
                sigma[p, k] = self.model.addVar(vtype=GRB.CONTINUOUS)
>>>>>>> 28c36e876a5bd08659c95b6afa54c4d5bb178180:python/models.py

        somme=0
        # Variables for utility at each breakpoint
<<<<<<< HEAD:python_folder/models.py
        breakpoint_utils={}
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L+1):
                    breakpoint_utils[k, i, b] = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"breakpoint_utils_{k}_{i}_{b}")
=======
        # breakpoint_utils = self.model.addVars(self.K, self.criterions, self.L+1, lb=0, ub=GRB.INFINITY, name="breakpoint_utils")
        self.breakpoint_utils= {}
        for k in range(self.K):
            for n in range(self.criterions):
                for l in range(self.L +1):
                    self.breakpoint_utils[k,n,l] = self.model.addVar(vtype=GRB.CONTINUOUS)
>>>>>>> 28c36e876a5bd08659c95b6afa54c4d5bb178180:python/models.py
        
        # Constraints for linear segments
        for k in range(self.K):
            for i in range(self.criterions):
                for b in range(self.L):
                    self.model.addConstr((self.breakpoint_utils[k, i, b+1] - self.breakpoint_utils[k, i, b]) >=0)

        # Function to calculate utility
        def calculate_utility(k, features):
            utility = 0
            for i, feature in enumerate(features):
                for b in range(self.L):
                    if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                        utility += self.breakpoint_utils[k, i, b] + ((self.breakpoint_utils[k, i, b+1]-self.breakpoint_utils[k, i, b])/(self.breakpoints[b+1]-self.breakpoints[b])) * (feature - self.breakpoints[b])
                        break
            return utility

        # Utility difference constraint
        for p in range(P):
            self.model.addConstr(sum(I[p, k] for k in range(self.K)) >= 1)
            for k in range(self.K):
                self.model.addConstr(M * (1 - I[p, k]) + calculate_utility(k, X[p]) - calculate_utility(k, Y[p]) - e +sigma[p,k]>= 0)
        for p in range(P):
            for k in range(self.K):
                somme+= sigma[p,k]
        self.model.setObjective(somme, GRB.MINIMIZE)
        self.breakpoint_utils = breakpoint_utils
        self.model.optimize()

        return

    def predict_utility(self, X):
        utilities = np.zeros((len(X), self.K))  # Tableau 2D: lignes pour les échantillons, colonnes pour les clusters
        for p in range(len(X)):
<<<<<<< HEAD:python_folder/models.py
            for k in range(self.K):
                utility = 0
                for i, feature in enumerate(X[p]):
                    for b in range(self.L):
                        if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                            # Calculer l'utilité pour chaque cluster séparément
                            utility += self.breakpoint_utils[k, i, b].X + ((self.breakpoint_utils[k, i, b+1].X - self.breakpoint_utils[k, i, b].X) / (self.breakpoints[b+1] - self.breakpoints[b])) * (feature - self.breakpoints[b])
                            break
                utilities[p, k] = utility  # Stocker l'utilité de l'échantillon 'p' pour le cluster 'k'
        return utilities




=======
            utility = []
            for k in range(self.K):
                score=0
                for i, feature in enumerate(X[p]):
                    for b in range(self.L):
                        if self.breakpoints[b] <= feature < self.breakpoints[b + 1]:
                            score += self.breakpoint_utils[k, i, b].X + ((self.breakpoint_utils[k, i, b+1].X-self.breakpoint_utils[k, i, b].X)/(self.breakpoints[b+1]-self.breakpoints[b])) * (feature - self.breakpoints[b])
                        
                            break
                utility.append(score)
            value.append(utility)               
        return np.stack(value, axis=0)
>>>>>>> 28c36e876a5bd08659c95b6afa54c4d5bb178180:python/models.py
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.



class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
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
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
