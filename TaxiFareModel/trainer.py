# imports
from TaxiFareModel import data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import stats


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


        # Features
        # Numerical features
        features_to_standard_scale = []
        features_to_minmax_scale = []
        for feature in X.select_dtypes('number').columns:
            statistic, pvalue = stats.shapiro(X[feature])
            if pvalue >= 0.95:
                features_to_standard_scale.append(feature)
            features_to_minmax_scale.append(feature)

        # Categorical features
        features_to_encode = [x for x in X.select_dtypes('object').columns]

        # Scaling
        # Standard scaling
        standard_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ])

        # MinMax scaling
        minmax_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', MinMaxScaler())
        ])

        # Preprocessing
        preprocessor = ColumnTransformer([
            ('standard_transformer', standard_transformer, features_to_standard_scale),
            ('minmax_transformer', minmax_transformer, features_to_minmax_scale),
            ('one_hot_encoder', OneHotEncoder(
                handle_unknown="ignore"), features_to_encode)
        ])

        self.pipeline_params = {
            "preprocessor": {
                "name": "preprocessing",
                "_class": preprocessor
            },
            "model": {
                "name": "linear",
                "_class": LinearRegression
            }
        }

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        preprocessor = self.pipeline_params["preprocessor"]
        model = self.pipeline_params["model"]

        self.pipeline = Pipeline([
            (preprocessor["name"], preprocessor["_class"]),
            (model["name"], model["_class"]())
        ])

        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        return self.pipeline.score(X_test, y_test)


if __name__ == "__main__":
    # get data
    raw_data = data.get_data()
    # clean data
    df = data.clean_data(raw_data)
    # set X and y
    target = "fare_amount"
    X = df.drop(columns=target)
    y = df[target]
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    model_ready = trainer.set_pipeline()
    model_trained = trainer.run()
    # evaluate
    model_eval = trainer.evaluate(X_test, y_test)
