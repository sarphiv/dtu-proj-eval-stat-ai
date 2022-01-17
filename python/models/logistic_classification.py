from validation.validation_model import ValidationModel
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticClassifier(ValidationModel):
    def __init__(self): 
        pass

    def train_predict(self, train_features, train_labels, test_features):
        #Standardize features
        train_mean = train_features.mean(axis = 0)
        train_std = train_features.std(axis = 0)
        
        train_features_standardized = (train_features - train_mean)/train_std
        test_features_standardized  = (test_features  - train_mean)/train_std


        #Fit logistic model
        m = LogisticRegression(multi_class='multinomial', max_iter=1000)
        m.fit(train_features_standardized, train_labels)
        
        #Get prediction
        pred_labels = m.predict(test_features_standardized)

        return pred_labels
