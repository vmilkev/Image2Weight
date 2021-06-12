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
    
    def __init__(self, target, tr_features, vl_features, method):
        # class constructor
        self.target = target
        self.tr_features = tr_features
        self.vl_features = vl_features
        self.meth = method
        self.hpar_flag = False

    def fit(self):
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
    
    def pred(self):
        if self.meth == 'lr':        # Linear Regression            
            return self.__lr_pred()
        elif self.meth == 'rr':      # Ridge Regression
            return self.__rr_pred()
        elif self.meth == 'la':      # Lasso Regression
            return self.__la_pred()
        elif self.meth == 'dt':      # Decission Tree Regressor
            return self.__dt_pred()
        elif self.meth == 'rf':      # Random Forest Regressor
            return self.__rf_pred()
        elif self.meth == 'ab':     # Ada Boost Regressor
            return self.__ab_pred()
        elif self.meth == 'sv':     # Support Vector Regressor
            return self.__sv_pred()
        else:
            return

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
        self.lr_model = regression.fit(self.tr_features, self.target)
        
        return self.lr_model.intercept_, self.lr_model.coef_;

    def __rr_fit(self):
        # Ridge fits a linear model with coefficients w = (w1, …, wp)
        
        scaler = StandardScaler()
        tr_features_std = scaler.fit_transform( self.tr_features )

        if self.hpar_flag:
            regression = Ridge( alpha = self.hpar )
        else:
            regression = Ridge( alpha = 0.5 )

        self.rr_model = regression.fit(tr_features_std, self.target)
        
        return self.rr_model.intercept_, self.rr_model.coef_;

    def __la_fit(self):
        # Ridge fits a linear model with coefficients w = (w1, …, wp)
        
        scaler = StandardScaler()
        tr_features_std = scaler.fit_transform( self.tr_features )

        if self.hpar_flag:
            regression = Lasso( alpha = self.hpar, max_iter = 2000 )
        else:
            regression = Lasso( alpha = 0.5, max_iter = 2000 )

        self.la_model = regression.fit(tr_features_std, self.target)
        
        return self.la_model.intercept_, self.la_model.coef_;

    def __dt_fit(self):
        # Decission tree regressor
        
        if self.hpar_flag:
            decitree = DecisionTreeRegressor( random_state = self.hpar )
        else:
            decitree = DecisionTreeRegressor( random_state=0 )
        
        self.dt_model = decitree.fit(self.tr_features, self.target)

    def __rf_fit(self):
        # Random forest regressor
        
        if self.hpar_flag:
            decitree = RandomForestRegressor( random_state = self.hpar, n_jobs=-1 )
        else:
            decitree = RandomForestRegressor( random_state=0, n_jobs=-1 )
        
        self.rf_model = decitree.fit(self.tr_features, self.target)

    def __ab_fit(self):
        # Ada boost regressor
        
        if self.hpar_flag:
            decitree = AdaBoostRegressor( random_state = self.hpar )
        else:
            decitree = AdaBoostRegressor( random_state=0 )
        
        self.ab_model = decitree.fit(self.tr_features, self.target)

    def __sv_fit(self):
        # Support vector regressor
        
        if self.hpar_flag:
            suvec = svm.SVR( C = self.hpar )
        else:
            suvec = svm.SVR( C = 100.0 )
        
        self.sv_model = suvec.fit(self.tr_features, self.target)

#-----------------------------------------------------------------------------------------------

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

    def __la_pred(self):
        # Predict using the ridge regression model        
        try:
            self.la_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.la_model.predict(self.vl_features)

    def __dt_pred(self):
        # Predict using the decission tree regressor model        
        try:
            self.dt_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.dt_model.predict(self.vl_features)

    def __rf_pred(self):
        # Predict using the random forest regressor model        
        try:
            self.rf_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.rf_model.predict(self.vl_features)

    def __ab_pred(self):
        # Predict using the random forest regressor model        
        try:
            self.ab_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.ab_model.predict(self.vl_features)

    def __sv_pred(self):
        # Predict using the random forest regressor model        
        try:
            self.sv_model
        except NameError:
            print(f'\tThe model is not trained!')
        
        return self.sv_model.predict(self.vl_features)

#-----------------------------------------------------------------------------------------------

