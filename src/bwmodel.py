import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm

from sklearn.preprocessing import StandardScaler

class BWModel:
    
    def __init__(self):
        # class constructor
        self.hpar_flag = False

    def fit(self, target, tr_features, method):

        self.target = target
        self.tr_features = tr_features
        self.meth = method

        if self.meth == 'lr':       # Linear Regression            
            return self.__lr_fit()
        elif self.meth == 'rr':     # Ridge Regression
            return self.__rr_fit()
        elif self.meth == 'la':     # Lasso Regression
            return self.__la_fit()
        elif self.meth == 'dt':     # Decission Tree Regressor
            return self.__dt_fit()
        elif self.meth == 'rf':     # Random Forest Regressor
            return self.__rf_fit()
        elif self.meth == 'ab':     # Ada Boost Regressor
            return self.__ab_fit()
        elif self.meth == 'sv':     # Support Vector Regressor
            return self.__sv_fit()
        else:
            return
    
    def pred(self, vl_features):
        return self.__pred( vl_features )

    def set_hpar( self,val ):
        # setting hyperparameter is necessary
        self.hpar = val
        self.hpar_flag = True

    def rem_hpar( self ):
        # unsetting hyperparameter is necessary
        self.hpar_flag = False

#-----------------------------------------------------------------------------------------------

    def __lr_fit(self):
        # LinearRegression fits a linear model with coefficients w = (w1, …, wp)
        # to minimize the residual sum of squares between the observed targets in the dataset,
        # and the targets predicted by the linear approximation
        
        regression = LinearRegression()
        self.model = regression.fit(self.tr_features, self.target)
        
        #return self.lr_model.intercept_, self.lr_model.coef_;
        return self.model

    def __rr_fit(self):
        # Ridge fits a linear model with coefficients w = (w1, …, wp)
        
        scaler = StandardScaler()
        tr_features_std = scaler.fit_transform( self.tr_features )

        if self.hpar_flag:
            regression = Ridge( alpha = self.hpar )
        else:
            regression = Ridge( alpha = 0.5 )

        self.model = regression.fit(tr_features_std, self.target)
        
        return self.model

    def __la_fit(self):
        # Lasso fits a linear model with coefficients w = (w1, …, wp)
        
        scaler = StandardScaler()
        tr_features_std = scaler.fit_transform( self.tr_features )

        if self.hpar_flag:
            regression = Lasso( alpha = self.hpar, max_iter = 2000 )
        else:
            regression = Lasso( alpha = 0.5, max_iter = 2000 )

        self.model = regression.fit(tr_features_std, self.target)
        
        return self.model

    def __dt_fit(self):
        # Decission tree regressor
        
        if self.hpar_flag:
            decitree = DecisionTreeRegressor( random_state = self.hpar )
        else:
            decitree = DecisionTreeRegressor( random_state=0 )
        
        self.model = decitree.fit(self.tr_features, self.target)

        return self.model

    def __rf_fit(self):
        # Random forest regressor
        
        if self.hpar_flag:
            decitree = RandomForestRegressor( random_state = self.hpar, n_jobs=-1 )
        else:
            decitree = RandomForestRegressor( random_state=0, n_jobs=-1 )
        
        self.model = decitree.fit(self.tr_features, self.target)

        return self.model

    def __ab_fit(self):
        # Ada boost regressor
        
        if self.hpar_flag:
            decitree = AdaBoostRegressor( random_state = self.hpar )
        else:
            decitree = AdaBoostRegressor( random_state=0 )
        
        self.model = decitree.fit(self.tr_features, self.target)

        return self.model

    def __sv_fit(self):
        # Support vector regressor

        if self.hpar_flag:
            suvec = svm.SVR( C = self.hpar )
        else:
            suvec = svm.SVR( C = 10.0 )
        
        self.model = suvec.fit(self.tr_features, self.target)

        return self.model

#-----------------------------------------------------------------------------------------------

    def __pred(self, vl_features):
        # Predict using the linear model        
        try:
            self.model
        except NameError:
            print(f'\tThe model is not trained!')
                
        return self.model.predict(vl_features)

#-----------------------------------------------------------------------------------------------

