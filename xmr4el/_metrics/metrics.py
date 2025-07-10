import pandas as pd
import numpy as np

class Metrics():
    
    @staticmethod
    def test(path):
        
        df = pd.read_csv(path, sep='\t')
        
        print(df)
    
    @staticmethod
    def precison(y_true, y_pred, k=5):
        intersection = np.intersection1d(y_true, y_pred[:k])
        return len(intersection) / k
    
    @staticmethod
    def recall(y_true, y_pred, k=5):
        intersection = np.intersection1d(y_true, y_pred[:k])
        return len(intersection) / len(y_true)
        
        