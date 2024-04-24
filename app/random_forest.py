import numpy as np
import pandas as pd

import feature_encoders
import data_normalizers
import feature_selectors
import dimension_reduction_methods

from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score


def hyper_opt_func(params, x, y):
    """
    This function performs the hyperparameter optimisation
    """

    rf = RandomForestRegressor(
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        min_samples_split=params['min_samples_split'],
        n_estimators=params['n_estimators']
    )

    score = cross_val_score(rf, x, y, scoring='r2', cv=5).mean()

    return {'loss': -score, 'status': STATUS_OK}


class RandomForest:
    """
    This is the implementation of the Random Forest
    machine learning model.
    """

    def __init__(self):

        # model parameters
        self.n_estimator = 25
        self.max_depth = 30
        self.min_samples_leaf = 3
        self.min_samples_split = 2

        # model itself
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimator,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split
        )

        # model data
        self.training_data = None
        self.testing_data = None
        self.querying_data = None

        # user input parameters
        self.feature_encoding_method = None
        self.kmer_size = None
        self.data_normalization_algorithm = None
        self.feature_selection_algorithm = None
        self.feature_number = None
        self.hyper_opt_iterations = None
        self.model_number = None

        # training output statistics
        self.training_RMSE = None
        self.training_R_squared = None
        self.training_MAE = None
        self.training_percentage_2fold_error = None
        self.training_2fold_error = None
        self.training_RMSE_std = None
        self.training_R_squared_std = None
        self.training_MAE_std = None
        self.training_percentage_2fold_error_std = None
        self.training_2fold_error_std = None

        # testing output statistics
        self.testing_RMSE = None
        self.testing_R_squared = None
        self.testing_MAE = None
        self.testing_percentage_2fold_error = None
        self.testing_2fold_error = None

        # unsupervised learning
        self.dimensionality_reduction_algorithm = None
        self.dimension_number = 2

        # track progress (for users)
        self.trained_model = False
        self.tested_model = False
        self.queried_model = False

        # question answers
        self.use_feature_select = None
        self.use_unsupervised = None
        self.use_hyper_opt = None

        # temporary outputs
        self.encoded_train = None
        self.encoded_test = None
        self.encoded_query = None
        self.normalized_train = None
        self.normalized_test = None
        self.normalized_query = None
        self.selected_train = None
        self.selected_test = None
        self.selected_query = None
        self.selected_features = None
        self.unsupervised_train = None
        self.unsupervised_test = None
        self.unsupervised_query = None
        self.model_predictions = None
        self.query_predictions = None
        self.model_query_created_file = None

        # normalizers
        self.z_score_feature_normaliser = None
        self.z_score_target_normaliser = None
        self.min_max_feature_normaliser = None
        self.min_max_target_normaliser = None

        # files used
        self.training_file = None
        self.testing_file = None
        self.querying_file = None

    # ------------------------ SETTERS ------------------------ #

    def set_training_data(self, training_data):
        self.training_data = training_data

    def set_testing_data(self, testing_data):
        self.testing_data = testing_data

    def set_querying_data(self, querying_data):
        self.querying_data = querying_data

    def set_training_file(self, training_file):
        self.training_file = training_file

    def set_testing_file(self, testing_file):
        self.testing_file = testing_file

    def set_querying_file(self, querying_file):
        self.querying_file = querying_file

    def set_feature_encoding_method(self, method):
        self.feature_encoding_method = method

    def set_kmer_size(self, kmer_size):
        self.kmer_size = kmer_size

    def set_data_normalization_algorithm(self, algorithm):
        self.data_normalization_algorithm = algorithm

    def set_feature_selection_algorithm(self, algorithm):
        self.feature_selection_algorithm = algorithm

    def set_feature_number(self, number):
        self.feature_number = number

    def set_hyper_opt_iterations(self, iterations):
        self.hyper_opt_iterations = iterations

    def set_dimensionality_reduction_algorithm(self, algorithm):
        self.dimensionality_reduction_algorithm = algorithm

    def set_use_unsupervised(self, answer):
        self.use_unsupervised = answer

    def set_use_feature_select(self, answer):
        self.use_feature_select = answer

    def set_use_hyperopt(self, answer):
        self.use_hyper_opt = answer

    def set_model_num(self, number):
        self.model_number = f"model_{number}"

    # ------------------------ METHODS ------------------------ #

    def encode_features(self):
        """
        Method for applying the correct feature encoding method based on
        the user inputs
        """

        if self.feature_encoding_method == 'binary':
            self.encoded_train, self.encoded_test, self.encoded_query = \
                feature_encoders.encode_one_hot(self.training_data, self.testing_data, self.querying_data)

        elif self.feature_encoding_method == 'kmer':
            self.encoded_train, self.encoded_test, self.encoded_query = \
                feature_encoders.encode_kmer(self.training_data, self.testing_data, self.querying_data, self.kmer_size)

    def normalize_features(self):
        """
        Method for applying the correct feature normalization method based on
        the user inputs
        """

        if self.data_normalization_algorithm == 'zscore':
            self.normalized_train, self.normalized_test, self.z_score_feature_normaliser, \
                self.z_score_target_normaliser = \
                data_normalizers.z_score_normalization(self.encoded_train, self.encoded_test)
            if self.encoded_query is not None:
                self.normalized_query = self.z_score_feature_normaliser.transform(self.encoded_query)

        elif self.data_normalization_algorithm == 'minmax':
            self.normalized_train, self.normalized_test, self.min_max_feature_normaliser, \
                self.min_max_target_normaliser = \
                data_normalizers.min_max_normalization(self.encoded_train, self.encoded_test)
            if self.encoded_query is not None:
                self.normalized_query = self.min_max_feature_normaliser.transform(self.encoded_query)

    def train_model(self):
        """
        Method for training the machine learning model
        on the user-uploaded training data
        """

        # Encode and normalize the features in the uploaded data
        self.encode_features()
        self.normalize_features()

        # Apply dimensionality reduction (if unsupervised learning is enabled)
        if self.use_unsupervised == "yes":
            # PCA
            if self.dimensionality_reduction_algorithm == 'PCA':
                self.unsupervised_train, self.unsupervised_test, self.unsupervised_query = \
                    dimension_reduction_methods.use_pca(self.normalized_train, self.normalized_test,
                                                        self.normalized_query)

            # UMAP
            elif self.dimensionality_reduction_algorithm == 'UMAP':
                self.unsupervised_train, self.unsupervised_test, self.unsupervised_query = \
                    dimension_reduction_methods.use_umap(self.normalized_train, self.normalized_test,
                                                         self.normalized_query)

            # t-SNE
            elif self.dimensionality_reduction_algorithm == 't-SNE':
                self.unsupervised_train, self.unsupervised_test, self.unsupervised_query = \
                    dimension_reduction_methods.use_tsne(self.normalized_train, self.normalized_test,
                                                         self.normalized_query)

        # Apply feature selection (if enabled)
        if self.use_feature_select == "yes":
            # f-score
            if self.feature_selection_algorithm == "F-score":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    feature_selectors.f_score_selection(self.normalized_train, self.normalized_test,
                                                        self.normalized_query, self.feature_number)

            # weight importance
            elif self.feature_selection_algorithm == "Weight Importance":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    feature_selectors.weight_importance_selection(self.model, self.normalized_train,
                                                                  self.normalized_test,
                                                                  self.normalized_query, self.feature_number)

            # mutual information
            elif self.feature_selection_algorithm == "Mutual Information":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    feature_selectors.mutual_information_selection(self.normalized_train, self.normalized_test,
                                                                   self.normalized_query, self.feature_number)

            # PCA
            elif self.feature_selection_algorithm == "PCA":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    feature_selectors.pca_selection(self.normalized_train, self.normalized_test, self.normalized_query,
                                                    self.feature_number)

        # Prepare the training data
        x_train = self.selected_train if self.selected_train is not None else \
            self.normalized_train.drop('protein', axis=1)
        y_train = self.normalized_train['protein']

        # Apply hyperparameter optimisation (if enabled)
        if self.use_hyper_opt == "yes":
            # set up the parameter space for hyper-opt
            space = {
                'max_depth': hp.choice('max_depth',
                                       [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                'min_samples_split': hp.choice('min_samples_split', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                'n_estimators': hp.choice('n_estimators', [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
            }

            # Initialize trials object to store details of each iteration
            trials = Trials()

            # Create a partial function with X and y
            objective_with_data = partial(hyper_opt_func, x=x_train, y=y_train)

            # Run the optimizer
            best = fmin(fn=objective_with_data, space=space, algo=tpe.suggest, max_evals=self.hyper_opt_iterations,
                        trials=trials)

            # Select the best model
            self.model = RandomForestRegressor(
                max_depth=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100][best['max_depth']],
                min_samples_leaf=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][best['min_samples_leaf']],
                min_samples_split=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12][best['min_samples_split']],
                n_estimators=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95][best['n_estimators']]
            )

        # Initialize arrays to store metrics for each fold
        rmse_per_fold = []
        r2_per_fold = []
        mae_per_fold = []
        two_fold_error_per_fold = []
        percentage_2fold_error_per_fold = []

        # Setup K-Fold cross-validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Assuming x_train and y_train are numpy arrays, convert them to pandas DataFrame or Series
        x_train_df = pd.DataFrame(x_train)
        y_train_series = pd.Series(y_train)

        # Loop over each fold in the cross-validation
        for train_index, test_index in k_fold.split(x_train_df):
            # Split data into training and test sets for this fold using iloc
            x_train_fold = x_train_df.iloc[train_index]
            x_test_fold = x_train_df.iloc[test_index]
            y_train_fold = y_train_series.iloc[train_index]
            y_test_fold = y_train_series.iloc[test_index]

            # Fit the model on the training fold
            self.model.fit(x_train_fold, y_train_fold)

            # Make predictions on the test fold
            predictions = self.model.predict(x_test_fold)

            # Compute metrics for this fold and append to respective lists
            rmse_per_fold.append(np.sqrt(mean_squared_error(y_test_fold, predictions)))
            r2_per_fold.append(r2_score(y_test_fold, predictions))
            mae_per_fold.append(mean_absolute_error(y_test_fold, predictions))
            two_fold_error_per_fold.append(np.mean((predictions / y_test_fold <= 2) & (y_test_fold / predictions <= 2)))
            percentage_2fold_error_per_fold.append(
                np.mean((predictions / y_test_fold <= 2) & (y_test_fold / predictions <= 2)) * 100)

        # Calculate the average and standard deviation for each metric

        # Calculate RMSE
        self.training_RMSE = np.mean(rmse_per_fold)
        self.training_RMSE_std = np.std(rmse_per_fold)

        # Calculate R-Squared
        self.training_R_squared = np.mean(r2_per_fold)
        self.training_R_squared_std = np.std(r2_per_fold)

        # Calculate MAE
        self.training_MAE = np.mean(mae_per_fold)
        self.training_MAE_std = np.std(mae_per_fold)

        # Calculate Percentage within 2-Fold Error
        self.training_percentage_2fold_error = np.mean(percentage_2fold_error_per_fold)
        self.training_percentage_2fold_error_std = np.std(percentage_2fold_error_per_fold)

        # Calculate 2-fold error
        self.training_2fold_error = np.mean(two_fold_error_per_fold)
        self.training_2fold_error_std = np.std(two_fold_error_per_fold)

        # Retrain on the entire training dataset
        self.model.fit(x_train, y_train)

        self.trained_model = True

    def test_model(self):
        """
        Method for testing the machine learning model
        on the user-uploaded testing data
        """

        # Prepare the test data
        x_test = self.selected_test if self.selected_test is not None else self.normalized_test.drop('protein', axis=1)
        y_test = self.normalized_test['protein']

        # Get the scaler (to un-normalise the predictions)
        scaler = None
        if self.data_normalization_algorithm == 'zscore':
            scaler = self.z_score_target_normaliser

        elif self.data_normalization_algorithm == 'minmax':
            scaler = self.min_max_target_normaliser

        # Make predictions using the trained model
        test_predictions = self.model.predict(x_test)

        # Un-normalise the predictions
        self.model_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten().tolist()

        # Calculate RMSE for test data
        self.testing_RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

        # Calculate R-Squared for test data
        self.testing_R_squared = r2_score(y_test, test_predictions)

        # Calculate MAE for test data
        self.testing_MAE = mean_absolute_error(y_test, test_predictions)

        # Calculate Percentage within 2-Fold Error for test data
        self.testing_percentage_2fold_error = \
            np.mean((test_predictions / y_test <= 2) & (y_test / test_predictions <= 2)) * 100

        # Calculate 2-fold error
        self.testing_2fold_error = np.mean((test_predictions / y_test <= 2) & (y_test / test_predictions <= 2))

        self.tested_model = True

    def query_model(self):
        """
        Method for querying the machine learning model
        on the user-uploaded querying data
        """

        scaler = None
        if self.data_normalization_algorithm == 'zscore':
            scaler = self.z_score_target_normaliser

        elif self.data_normalization_algorithm == 'minmax':
            scaler = self.min_max_target_normaliser

        normalized_x_df = pd.DataFrame(self.normalized_query)
        if self.selected_query is not None:
            normalized_x_df = pd.DataFrame(self.selected_query)

        normalized_predictions = self.model.predict(normalized_x_df)

        # Convert predictions back to original scale
        predictions_original_scale = scaler.inverse_transform(normalized_predictions.reshape(-1, 1))

        # Convert predictions to a DataFrame
        self.query_predictions = pd.DataFrame(predictions_original_scale, columns=['protein'])

        # Concatenate the raw data DataFrame with the predictions DataFrame
        self.model_query_created_file = pd.concat([self.querying_data['sequence'], self.query_predictions], axis=1)

        self.queried_model = True
