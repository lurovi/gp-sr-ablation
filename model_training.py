#from pysr import PySRRegressor
from pyGPGOMEA import GPGOMEARegressor as GPG
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import random
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from data_loader import split_data
from utils import compute_linear_scaling


class CastFloatScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return np.asarray(X_scaled, dtype=np.float64)

    def get_params(self, deep=True):
        out = super().get_params(deep=deep)
        out.update({
            'scaler': self.scaler,
        })
        return out

    def set_params(self, **params):
        super().set_params(**params)

        for key in ['scaler']:
            if key in params:
                setattr(self, key, params[key])

        return self


class NumpyPassthroughScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)


class LinearScalingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline, linear_scaling):
        self.pipeline = pipeline
        self.linear_scaling = linear_scaling

        self.slope_ = 1.0
        self.intercept_ = 0.0

        self.VERY_LARGE_VAL = 1e+300

    def fit(self, X, y):
        X = X.copy().reset_index(drop=True)

        self.pipeline.fit(X, y)

        if self.linear_scaling:
            p = self.pipeline.predict(X)
            slope, intercept = compute_linear_scaling(y, p)
            self.slope_ = np.core.umath.clip(slope, -self.VERY_LARGE_VAL, self.VERY_LARGE_VAL)
            self.intercept_ = np.core.umath.clip(intercept, -self.VERY_LARGE_VAL, self.VERY_LARGE_VAL)

        return self

    def predict(self, X):
        return np.core.umath.clip(self.slope_ * self.pipeline.predict(X), -self.VERY_LARGE_VAL, self.VERY_LARGE_VAL) + self.intercept_

    def get_params(self, deep=True):
        out = super().get_params(deep=deep)
        out.update({
            'pipeline': self.pipeline,
            'linear_scaling': self.linear_scaling,
        })
        return out

    def set_params(self, **params):
        super().set_params(**params)

        for key in ['pipeline', 'linear_scaling']:
            if key in params:
                setattr(self, key, params[key])

        return self


def pysr_operator_dict():
    # Define the alias-to-PySR mapping dictionary
    return {
            'add': ('+', 'binary'),
            'sub': ('-', 'binary'),
            'mul': ('*', 'binary'),
            'div': ('/', 'binary'),
            'pow': ('^', 'binary'),
            'mod': ('mod', 'binary'),
            'max': ('max', 'binary'),
            'min': ('min', 'binary'),
            'sin': ('sin', 'unary'),
            'cos': ('cos', 'unary'),
            'tan': ('tan', 'unary'),
            'tanh': ('tanh', 'unary'),
            'relu': ('relu', 'unary'),
            'gamma': ('gamma', 'unary'),
            'sinc': ('sinc', 'unary'),
            'exp': ('exp', 'unary'),
            'log': ('log', 'unary'),
            'abs': ('abs', 'unary'),
            'sqrt': ('sqrt', 'unary'),
            'cbrt': ('cbrt', 'unary'),
            'square': ('square', 'unary'),
            'cube': ('cube', 'unary'),
            'sign': ('sign', 'unary'),
            'neg': ('neg', 'unary'),
            'inv': ('inv', 'unary'),
        }


def gp_operator_dict():
    # Define the alias-to-GP mapping dictionary
    return {
        'add': '+',
        'sub': '-',
        'mul': '*',
        'div': 'p/',
        'sin': 'sin',
        'cos': 'cos',
        'tanh': 'tanh',
        'exp': 'exp',
        'log': 'plog',
        'sqrt': 'sqrt',
        'square': '^2',
    }


def compute_binary_unary_operators_from_aliases(aliases):
    # Generate PySR operator strings
    d = pysr_operator_dict()
    binary_operators = [d[a][0] for a in aliases if d[a][1] == 'binary']
    unary_operators = [d[a][0] for a in aliases if d[a][1] == 'unary']
    # Output
    return binary_operators, unary_operators


def compute_operators_string_from_aliases(aliases):
    # Generate GP operator strings
    d = gp_operator_dict()
    operators = [d[a] for a in aliases]
    # Output
    return '_'.join(operators)


def preprocess_and_train(df, scaling_strategy, model_name, test_size, seed, functions, pop_size, num_gen, perc_train_records, tournament_size, linear_scaling, verbose=False):
    random.seed(seed)
    np.random.seed(seed)

    if model_name not in ('symbolic_regression', 'genetic_programming'):
        functions = []
        pop_size = 0
        num_gen = 0
        tournament_size = 0

    binary_operators, unary_operators = compute_binary_unary_operators_from_aliases(functions)
    str_operators = compute_operators_string_from_aliases(functions)

    target = df.columns[-1]
    df = df.copy()

    # Split data
    train_df, test_df = split_data(df, test_size=test_size, random_state=seed)

    X_train = train_df.iloc[:,:-1]
    y_train = train_df[target].to_numpy()

    if perc_train_records > 0.0:
        n_train_records = int(perc_train_records * len(X_train))
        if n_train_records <= 1:
            n_train_records = 2
        X_train = X_train.iloc[:n_train_records,:]
        y_train = y_train[:n_train_records]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df[target].to_numpy()

    if scaling_strategy == "standard":
        scaler = StandardScaler()
    elif scaling_strategy == "minmax":
        scaler = MinMaxScaler()
    elif scaling_strategy == "robust":
        scaler = RobustScaler()
    elif scaling_strategy == "none":
        scaler = NumpyPassthroughScaler()
    else:
        raise ValueError("Invalid scaling strategy. Choose 'standard', 'minmax', 'robust', or 'none'.")

    scaler = CastFloatScaler(scaler)

    # Model selection
    models = {
        "linear": LinearRegression(),
        "sgd": SGDRegressor(random_state=seed),
        "elasticnet": ElasticNet(random_state=seed, l1_ratio=0.5, alpha=1.0, selection='cyclic'),
        "random_forest": RandomForestRegressor(random_state=seed, n_estimators=100, max_depth=9),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed, n_estimators=100, max_depth=9),
        "bagging": BaggingRegressor(random_state=seed, n_estimators=100),
        "adaboost": AdaBoostRegressor(random_state=seed),
        "knn": KNeighborsRegressor(),
        "mlp": MLPRegressor(random_state=seed),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "svr": SVR(),
        "genetic_programming": GPG(
            time=-1,
            evaluations=-1,
            generations=num_gen,
            popsize=pop_size,
            functions=str_operators,
            tournament=tournament_size,
            prob='symbreg',
            multiobj=False,
            linearscaling=False,
            erc=True,
            classweights=False,
            gomea=False,
            gomfos='',
            subcross=0.5,
            submut=0.5,
            reproduction=0.0,
            sblibtype=False,
            sbrdo=0.0,
            sbagx=0.0,
            unifdepthvar=True,
            elitism=1,
            ims=False,
            syntuniqinit=1000,
            initmaxtreeheight=4,
            inittype=False,
            maxtreeheight=12,
            maxsize=40,
            validation=False,
            coeffmut=False,
            gomcoeffmutstrat=False,
            batchsize=False,
            seed=seed,
            parallel=1,
            caching=False,
            silent=True,
            logtofile=False
	    ),
        # "symbolic_regression": PySRRegressor(
        #     random_state=seed,
        #     parallelism='serial',
        #     deterministic=True,
        #     progress=False,
        #     verbosity=0,
        #     parsimony=0.0,
        #     maxsize=30,
        #     maxdepth=20,
        #     batching=False,
        #     model_selection='accuracy',
        #     tournament_selection_n=tournament_size,
        #     niterations=num_gen,  # GP generations
        #     population_size=pop_size,  # Classic GP population size
        #     ncycles_per_iteration=5,  # ≈ pop_size / 2 mutations per gen
        #     topn=1,  # Elitism: 1 elite kept per generation
        #     populations=1,  # No islands
        #     fast_cycle=True,
        #     crossover_probability=0.5,
        #     binary_operators=binary_operators,
        #     unary_operators=unary_operators,
        # ),
    }

    if model_name not in models:
        raise ValueError("Invalid model name")

    model = models[model_name]

    # Hyperparameter grids
    param_grids_0 = {
        "linear": {},
        "sgd": {
            "regressor__loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "regressor__penalty": ["l2", "l1", "elasticnet"],
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "regressor__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "regressor__learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
        },
        "elasticnet": {
            "regressor__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "regressor__alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "regressor__selection": ['cyclic', 'random'],
        },
        "random_forest": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200, 500, 1000],
            "regressor__oob_score": [True, False],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__max_features": ['sqrt', 'log2', None],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [0.001, 0.01, 0.1, 1, 2, 4],
            "regressor__criterion": ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'],
        },
        "gradient_boosting": {
            "regressor__loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
            "regressor__learning_rate": [1e-4, 1e-3, 1e-2, 1e-1],
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200, 500, 1000],
            "regressor__criterion": ['friedman_mse', 'squared_error'],
            "regressor__max_features": ['sqrt', 'log2'],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [0.001, 0.01, 0.1, 1, 2, 4],
            "regressor__validation_fraction": [0.1, 0.2, 0.3],
            "regressor__n_iter_no_change": [10, 20, 30],
        },
        "bagging": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200, 500, 1000],
            "regressor__oob_score": [True, False],
        },
        "adaboost": {
            "regressor__n_estimators": [10, 20, 50, 100, 150, 200, 500, 1000],
            "regressor__learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__loss": ["linear", "square", "exponential"],
        },
        "knn": {
            "regressor__n_neighbors": [3, 5, 7],
            "regressor__weights": ['uniform', 'distance'],
            "regressor__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        },
        "mlp": {
            "regressor__hidden_layer_sizes": [(50,), (20,), (50,50), (20,20), (50,50,50), (20,20,20)],
            "regressor__activation": ['tanh', 'relu', 'logistic'],
            "regressor__solver": ['adam'],
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "regressor__batch_size": [16, 32, 64, 'auto'],
            "regressor__learning_rate": ['constant', 'invscaling', 'adaptive'],
            "regressor__max_iter": [200, 400, 600],
            "regressor__validation_fraction": [0.1, 0.2, 0.3],
            "regressor__n_iter_no_change": [10, 20, 30],
            "regressor__early_stopping": [True, False],
        },
        "decision_tree": {
            "regressor__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "regressor__splitter": ["best", "random"],
            "regressor__max_depth": [3, 6, 9, 12, 15],
            "regressor__max_features": ['sqrt', 'log2'],
        },
        "svr": {
            "regressor__kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            "regressor__C": [0.01, 0.1, 0.5, 1, 10, 100, 1000],
            "regressor__gamma": [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0],
            "regressor__degree": [1, 2, 3],
            "regressor__epsilon": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 0.2, 0.3, 0.5],
            "regressor__coef0": [0.0, 0.1, 0.2, 0.3],
            "regressor__shrinking": [True, False],
            "regressor__max_iter": [200, 400, 600, -1]
        },
        "symbolic_regression": {},
        "genetic_programming": {},
    }

    param_grids = {k: {k1.replace('regressor__', 'pipeline__regressor__'): param_grids_0[k][k1] for k1 in param_grids_0[k]} for k in param_grids_0}

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', model)
    ])

    linear_scaling_regressor = LinearScalingRegressor(pipeline=pipeline, linear_scaling=linear_scaling)
    grid = param_grids[model_name]

    if len(grid) != 0:
        # Hyperparameter tuning
        try:
            search = RandomizedSearchCV(linear_scaling_regressor, grid, n_iter=20, cv=4, n_jobs=1,
                                        scoring='neg_root_mean_squared_error', random_state=seed, refit=True,
                                        verbose=verbose)
            search.fit(X_train, y_train)
            estimator = search.best_estimator_
        except ValueError:
            try:
                search = RandomizedSearchCV(linear_scaling_regressor, grid, n_iter=20, cv=2, n_jobs=1,
                                            scoring='neg_root_mean_squared_error', random_state=seed, refit=True,
                                            verbose=verbose)
                search.fit(X_train, y_train)
                estimator = search.best_estimator_
            except ValueError:
                try:
                    linear_scaling_regressor.fit(X_train, y_train)
                    estimator = linear_scaling_regressor
                except ValueError:
                    train_df, _ = split_data(df, test_size=test_size, random_state=seed)
                    X_train = train_df.iloc[:, :-1]
                    y_train = train_df[target].to_numpy()
                    linear_scaling_regressor.fit(X_train, y_train)
                    estimator = linear_scaling_regressor
    else:
        linear_scaling_regressor.fit(X_train, y_train)
        estimator = linear_scaling_regressor

    return estimator, X_train, X_test, y_train, y_test
