import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def min_max_normalization(train_data, test_data):
    """
    This function applies the MinMax feature normalization
    """

    # Initialize scaler
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Separate features and target
    train_features = train_data.drop('protein', axis=1)
    train_target = train_data['protein']
    test_features = test_data.drop('protein', axis=1)
    test_target = test_data['protein']

    # Normalize features
    train_features_scaled = feature_scaler.fit_transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)

    # Optionally, normalize target
    train_target_scaled = target_scaler.fit_transform(train_target.values.reshape(-1, 1))
    test_target_scaled = target_scaler.transform(test_target.values.reshape(-1, 1))

    # Convert scaled features back to DataFrame
    train_features_scaled_df = pd.DataFrame(train_features_scaled, columns=train_features.columns)
    test_features_scaled_df = pd.DataFrame(test_features_scaled, columns=test_features.columns)

    # Recombine scaled features with target
    train_final_scaled = pd.concat(
        [pd.DataFrame(train_target_scaled, columns=['protein']), train_features_scaled_df], axis=1)
    test_final_scaled = pd.concat(
        [pd.DataFrame(test_target_scaled, columns=['protein']), test_features_scaled_df], axis=1)

    return train_final_scaled, test_final_scaled


def z_score_normalization(train_data, test_data):
    """
    This function applies the Z-score feature normalization
    """

    # Initialize scaler
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Separate features and target
    train_features = train_data.drop('protein', axis=1)
    train_target = train_data['protein']
    test_features = test_data.drop('protein', axis=1)
    test_target = test_data['protein']

    # Standardize features
    train_features_standardized = feature_scaler.fit_transform(train_features)
    test_features_standardized = feature_scaler.transform(test_features)

    # Optionally, standardize target
    train_target_standardized = target_scaler.fit_transform(train_target.values.reshape(-1, 1))
    test_target_standardized = target_scaler.transform(test_target.values.reshape(-1, 1))

    # Convert standardized features back to DataFrame
    train_features_standardized_df = pd.DataFrame(train_features_standardized, columns=train_features.columns)
    test_features_standardized_df = pd.DataFrame(test_features_standardized, columns=test_features.columns)

    # Recombine standardized features with target
    train_final_standardized = pd.concat(
        [pd.DataFrame(train_target_standardized, columns=['protein']), train_features_standardized_df], axis=1)
    test_final_standardized = pd.concat(
        [pd.DataFrame(test_target_standardized, columns=['protein']), test_features_standardized_df], axis=1)

    return train_final_standardized, test_final_standardized
