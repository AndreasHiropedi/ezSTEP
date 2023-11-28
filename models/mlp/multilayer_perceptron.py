import numpy as np
from models.features import encoders
from models.features import normalizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor


class MultiLayerPerceptron:
    """
    This is the implementation of the Multi-layer Perceptron
    machine learning model.
    """

    def __init__(self):
        # model parameters
        self.hidden_layer_number = 3
        self.hidden_layer_size = 100
        self.max_iter = 25

        # model itself
        self.model = MLPRegressor(
            activation='relu',
            alpha=0.0001,
            batch_size='auto',
            beta_1=0.9,
            beta_2=0.999,
            early_stopping=True,
            epsilon=1e-08,
            hidden_layer_sizes=tuple(self.hidden_layer_size for _ in range(self.hidden_layer_number)),
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=self.max_iter,
            momentum=0.9,
            nesterovs_momentum=True,
            power_t=0.5,
            random_state=None,
            shuffle=True,
            solver='adam',
            tol=1e-5,
            validation_fraction=0.1,
            verbose=False,
            warm_start=False
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

        # training output statistics
        self.training_RMSE = None
        self.training_R_squared = None
        self.training_MAE = None
        self.training_percentage_2fold_error = None

        # testing output statistics
        self.testing_RMSE = None
        self.testing_R_squared = None
        self.testing_MAE = None
        self.testing_percentage_2fold_error = None

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
        self.normalized_train = None
        self.normalized_test = None

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

    # ------------------------ METHODS ------------------------ #

    def encode_features(self):
        """
        Method for applying the correct feature encoding method based on
        the user inputs
        """

        if self.feature_encoding_method == 'binary':
            self.encoded_train, self.encoded_test = encoders.encode_one_hot(self.training_data, self.testing_data)

        elif self.feature_encoding_method == 'kmer':
            self.encoded_train, self.encoded_test = \
                encoders.encode_kmer(self.training_data, self.testing_data, self.kmer_size)

    def normalize_features(self):
        """
        Method for applying the correct feature normalization method based on
        the user inputs
        """

        if self.feature_normalization_algorithm == 'zscore':
            self.normalized_train, self.normalized_test = \
                normalizers.z_score_normalization(self.encoded_train, self.encoded_test)

        elif self.feature_normalization_algorithm == 'minmax':
            self.normalized_train, self.normalized_test = \
                normalizers.min_max_normalization(self.encoded_train, self.encoded_test)

    def train_model(self):
        """
        Method for training the machine learning model
        on the user-uploaded training data
        """

        # Encode and normalize the features in the uploaded data
        self.encode_features()
        self.normalize_features()

        # Prepare the training data
        x_train = self.normalized_train.drop('protein', axis=1)
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

        # Train the SVR model on the entire training set
        self.model.fit(x_train, y_train)

        self.trained_model = True

    def test_model(self):
        """
        Method for testing the machine learning model
        on the user-uploaded testing data
        """

        # Prepare the test data
        x_test = self.normalized_test.drop('protein', axis=1)
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

        self.tested_model = True

    def query_model(self):
        """
        Method for querying the machine learning model
        on the user-uploaded querying data
        """

        self.queried_model = True
