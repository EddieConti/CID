import numpy as np
import pandas as pd
from typing import Union
from sklearn.preprocessing import StandardScaler,OrdinalEncoder


class Dataset:
    """
    Function to prepare the dataset in a common pipeline.
    """

    def __init__(self,
                data : Union[pd.DataFrame, np.ndarray],
                features_names : list,
                numerical_features : list,
                target_feature : str,
                preprocess: bool,
    ):

        """
        :param training_data: A pandas dataframe or numpy array consisting of the data
        :param features_names: The list of features. Please provide the full list of features
        :param numerical_features: The list of numerical features in the dataset. From these
        we infer the categorical features.
        :param target_feature: The feature we want to predict
        :param preprocess: If to preprocess the dataset. Standardscaler and OrdinalEncoder are employed
        """
        assert isinstance(data, (pd.DataFrame, np.ndarray)), "data must be a pandas dataframe or a numpy array."
        
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data, columns = features_names)


        self.columns = self.data.columns
        self.numerical_features = numerical_features
        self.target_feature = target_feature
        self.amount_of_instances = len(self.data)

        self.categorical = [c for c in self.columnsif c not in self.numerical_features 
                            and c != self.target_feature]
                        
        if preprocess:

            scaler_continuous = StandardScaler()

            self.data[self.numerical_features] = scaler_continuous.fit_transform(
                self.data[self.numerical_features]
            )

            scaler_categorical = OrdinalEncoder()

            self.data[self.categorical] = scaler_categorical.fit_transform(
                self.data[self.categorical]
            )


@property
def get_medians(self) -> dict:
    """
    Extract the median for the continuous features 
    """
    
    return {column : np.median(column) for column in self.numerical_featres} 

@property
def get_means(self) -> dict:
    """
    Extract the mean for the continuous features 
    """

    return {column : np.mean(column) for column in self.numerical_featres} 

@property
def get_variance(self) -> dict:
    """
    Extract the variance for the continuous features 
    """
    return {column : np.var(column) for column in self.numerical_featres} 

