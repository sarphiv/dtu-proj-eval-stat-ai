from validation.validation_model import ValidationModel
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNNClassifier(ValidationModel):
    def __init__(self, k, person_weight): 
        self.k = k
        self.person_weight = person_weight


    def train_predict(self, train_features, train_labels, test_features):

        #WARN: Hardcoded spatial dimension size
        
        #Get spacial features
        X_train_spatial = train_features[:, 10:]
        X_test_spatial  = test_features[:, 10:]
        
        #Standardize spatial features
        X_train_mean = X_train_spatial.mean(axis = 0)
        X_train_std  = X_train_spatial.std(axis = 0)
        
        X_train_spatial_stand = (X_train_spatial - X_train_mean)/X_train_std
        X_test_spatial_stand  = (X_test_spatial  - X_train_mean)/X_train_std
        
        #Weigh person feature and create final features
        X_train = np.hstack((train_features[:, :10] * self.person_weight, X_train_spatial_stand))
        X_test  = np.hstack((test_features[:, :10]  * self.person_weight, X_test_spatial_stand))


        #Fit k nearest neighbor model
        m = KNeighborsClassifier(self.k)
        m.fit(X_train, train_labels)
        
        #Get prediction
        pred_labels = m.predict(X_test)
        
        return pred_labels
