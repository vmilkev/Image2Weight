import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class BWModel:
    
    def __init__(self, target, tr_features, vl_features, method):
        # class constructor
        self.target = target
        self.tr_features = tr_features
        self.vl_features = vl_features
        self.meth = method

    def fit(self):
        if self.meth == 'lr':             # Linear Regression            
            return self.__lr_fit()
        elif self.meth == 'rr':           # Ridge Regression
            return self.__rr_fit()
        # elif method == 'cls':
        #     return self.__cluster_s(data, icol1, icol2, threshold1, 0)
        # elif method == 'zs/cls':
        #     return self.__zscorecluster_s(data, icol1, icol2, threshold1, threshold2)
        else:
            return
    
    def pred(self):
        if self.meth == 'lr':             # Linear Regression            
            return self.__lr_pred()
        elif self.meth == 'rr':           # Ridge Regression
            return self.__rr_pred()
        # elif method == 'cls':
        #     return self.__cluster_s(data, icol1, icol2, threshold1, 0)
        # elif method == 'zs/cls':
        #     return self.__zscorecluster_s(data, icol1, icol2, threshold1, threshold2)
        else:
            return

    def __lr_fit(self):
        # LinearRegression fits a linear model with coefficients w = (w1, …, wp)
        # to minimize the residual sum of squares between the observed targets in the dataset,
        # and the targets predicted by the linear approximation
        
        regression = LinearRegression()
        self.lr_model = regression.fit(self.tr_features, self.target)
        
        return self.lr_model.intercept_, self.lr_model.coef_;

    def __rr_fit(self):
        # Ridge fits a linear model with coefficients w = (w1, …, wp)
        
        scaler = StandardScaler()
        tr_features_std = scaler.fit_transform( self.tr_features )
        regression = Ridge( alpha = 0.5 )
        self.rr_model = regression.fit(tr_features_std, self.target)
        
        return self.rr_model.intercept_, self.rr_model.coef_;

    def __lr_pred(self):
        # Predict using the linear model        
        try:
            self.lr_model
        except NameError:
            print(f'\tThe model is not trained!')
                
        return self.lr_model.predict(self.vl_features)

    def __rr_pred(self):
        # Predict using the ridge regression model        
        try:
            self.rr_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.rr_model.predict(self.vl_features)

