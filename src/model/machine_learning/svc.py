from .base_model import BaseModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pickle

class Svc(BaseModel):
    def __init__(self, **svc_params):
        self.model = make_pipeline(
                #Standard Scaler trasforma tutte le feature per avere : media=0, dev_standard=1. Questo migliora la stabilità e le prestazioni del modello SVM
                StandardScaler(), #standardizza le feature
                SVC(**svc_params))
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Modello SVC addestrato con successo.")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)
        
    def confusion_matrix(self, y_test, y_pred):
        return confusion_matrix(y_test, y_pred)

    
    def cross_validation(self, X_train, y_train, folders):
        return cross_val_score(self.model, X_train, y_train, cv=folders)
        