import numpy as np
import pandas as pd

from models.features import encoders
from models.features import normalizers
from models.features import selectors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold


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
        self.feature_normalization_algorithm = None
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

    def set_feature_normalization_algorithm(self, algorithm):
        self.feature_normalization_algorithm = algorithm

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
                encoders.encode_one_hot(self.training_data, self.testing_data, self.querying_data)

        elif self.feature_encoding_method == 'kmer':
            self.encoded_train, self.encoded_test, self.encoded_query = \
                encoders.encode_kmer(self.training_data, self.testing_data, self.querying_data, self.kmer_size)

    def normalize_features(self):
        """
        Method for applying the correct feature normalization method based on
        the user inputs
        """

        if self.feature_normalization_algorithm == 'zscore':
            self.normalized_train, self.normalized_test, self.z_score_feature_normaliser, \
                self.z_score_target_normaliser = \
                normalizers.z_score_normalization(self.encoded_train, self.encoded_test)
            if self.encoded_query is not None:
                self.normalized_query = self.z_score_feature_normaliser.transform(self.encoded_query)

        elif self.feature_normalization_algorithm == 'minmax':
            self.normalized_train, self.normalized_test, self.min_max_feature_normaliser, \
                self.min_max_target_normaliser = \
                normalizers.min_max_normalization(self.encoded_train, self.encoded_test)
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

        # Apply feature selection (if enabled)
        if self.use_feature_select == "yes":
            # f-score
            if self.feature_selection_algorithm == "f-score":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    selectors.f_score_selection(self.normalized_train, self.normalized_test,
                                                self.normalized_query, self.feature_number)

            # weight importance
            elif self.feature_selection_algorithm == "weight":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    selectors.weight_importance_selection(self.model, self.normalized_train, self.normalized_test,
                                                          self.normalized_query, self.feature_number)

            # mutual information
            elif self.feature_selection_algorithm == "mutual":
                self.selected_train, self.selected_test, self.selected_query, self.selected_features = \
                    selectors.mutual_information_selection(self.normalized_train, self.normalized_test,
                                                           self.normalized_query, self.feature_number)

        # Prepare the training data
        x_train = self.selected_train if self.selected_train is not None else \
            self.normalized_train.drop('protein', axis=1)
        y_train = self.normalized_train['protein']

        # Setup K-Fold cross-validation
        k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Get cross-validated predictions
        predictions = cross_val_predict(self.model, x_train, y_train, cv=k_fold)

        # Calculate RMSE
        self.training_RMSE = np.sqrt(mean_squared_error(y_train, predictions))

        # Calculate R-Squared
        self.training_R_squared = r2_score(y_train, predictions)

        # Calculate MAE
        self.training_MAE = mean_absolute_error(y_train, predictions)

        # Calculate Percentage within 2-Fold Error
        self.training_percentage_2fold_error = \
            np.mean((predictions / y_train <= 2) & (y_train / predictions <= 2)) * 100

        # Calculate 2-fold error
        self.training_2fold_error = np.mean((predictions / y_train <= 2) & (y_train / predictions <= 2))

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

        # Make predictions using the trained model
        test_predictions = self.model.predict(x_test)

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
        if self.feature_normalization_algorithm == 'zscore':
            scaler = self.z_score_target_normaliser

        elif self.feature_normalization_algorithm == 'minmax':
            scaler = self.min_max_target_normaliser

        normalized_x_df = pd.DataFrame(self.normalized_query, columns=self.encoded_query.columns)
        if self.selected_query is not None:
            normalized_x_df = pd.DataFrame(self.selected_query,
                                           columns=self.encoded_query.columns[self.selected_features])

        normalized_predictions = self.model.predict(normalized_x_df)

        # Convert predictions back to original scale
        predictions_original_scale = scaler.inverse_transform(normalized_predictions.reshape(-1, 1))

        # Convert predictions to a DataFrame
        predictions_df = pd.DataFrame(predictions_original_scale, columns=['protein'])

        # Concatenate the raw data DataFrame with the predictions DataFrame
        combined_df = pd.concat([self.querying_data['sequence'], predictions_df], axis=1)

        # Write to a CSV file
        combined_df.to_csv(f'{self.model_number}_query_data_predictions.csv', index=False)

        self.queried_model = True
