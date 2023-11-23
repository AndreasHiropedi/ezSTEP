

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
        self.print_res = False

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

        # output statistics
        self.RMSE = None
        self.R_squared = None
        self.MAE = None
        self.Percentage_2fold_error = None

        # unsupervised learning
        self.dimensionality_reduction_algorithm = None
        self.dimension_number = None

    # ------------------------ SETTERS ------------------------ #

    def set_training_data(self, training_data):
        self.training_data = training_data

    def set_testing_data(self, testing_data):
        self.testing_data = testing_data

    def set_querying_data(self, querying_data):
        self.querying_data = querying_data

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

    def set_dimension_number(self, number):
        self.dimension_number = number

    # ------------------------ METHODS ------------------------ #

    def train_model(self):
        """
        Method for training the machine learning model
        on the user-uploaded training data
        """
        pass

    def test_model(self):
        """
        Method for testing the machine learning model
        on the user-uploaded testing data
        """
        pass

    def query_model(self):
        """
        Method for querying the machine learning model
        on the user-uploaded querying data
        """
        pass

    def feature_selection(self):
        """
        Method for running the user-selected feature selection
        algorithm with the user-selected number of features
        """
        pass

    def hyper_opt(self):
        """
        Method for running the hyperparameter optimisation
        function using user input
        """
        pass
