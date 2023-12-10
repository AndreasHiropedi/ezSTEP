import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression, SelectFromModel


def f_score_selection(train_data, test_data, query_data, features_number):
    """
    This function implements the F-score feature selection algorithm.
    """

    # Separate features and target
    train_features = train_data.drop('protein', axis=1)
    train_target = train_data['protein']
    test_features = test_data.drop('protein', axis=1)

    # Create and fit SelectKBest with f-score for regression
    selector = SelectKBest(f_regression, k=features_number)
    train_selected = selector.fit_transform(train_features, train_target)

    # Transform the test data set
    test_selected = selector.transform(test_features)

    # Use the get_support method to get the mask of the selected features
    selected_features_mask = selector.get_support()

    # Transform query set (if applicable)
    if query_data is not None:
        query_selected = selector.transform(query_data)
        return train_selected, test_selected, query_selected, selected_features_mask

    return train_selected, test_selected, None, None


def weight_importance_selection(model, train_data, test_data, query_data, features_number):
    """
    This function implements the Weight Importance feature selection algorithm.
    """

    # Separate features and target
    train_features = train_data.drop('protein', axis=1)
    train_target = train_data['protein']
    test_features = test_data.drop('protein', axis=1)

    # Create and fit SelectFromModel with f-score for regression
    selector = SelectFromModel(model, threshold=-np.inf, max_features=features_number)
    train_selected = selector.fit_transform(train_features, train_target)

    # Transform the test data set
    test_selected = selector.transform(test_features)

    # Use the get_support method to get the mask of the selected features
    selected_features_mask = selector.get_support()

    # Transform query set (if applicable)
    if query_data is not None:
        query_selected = selector.transform(query_data)
        return train_selected, test_selected, query_selected, selected_features_mask

    return train_selected, test_selected, None, None


def mutual_information_selection(train_data, test_data, query_data, features_number):
    """
    This function implements the Mutual Information feature selection algorithm.
    """

    # Separate features and target
    train_features = train_data.drop('protein', axis=1)
    train_target = train_data['protein']
    test_features = test_data.drop('protein', axis=1)

    # Create and fit SelectKBest with mutual information for regression
    selector = SelectKBest(mutual_info_regression, k=features_number)
    train_selected = selector.fit_transform(train_features, train_target)

    # Transform the test data set
    test_selected = selector.transform(test_features)

    # Use the get_support method to get the mask of the selected features
    selected_features_mask = selector.get_support()

    # Transform query set (if applicable)
    if query_data is not None:
        query_selected = selector.transform(query_data)
        return train_selected, test_selected, query_selected, selected_features_mask

    return train_selected, test_selected, None, None
