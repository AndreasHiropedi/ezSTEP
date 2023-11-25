import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


def map_sequence(sequence, sequence_mapping, unknown_label):
    """
    This function maps sequences to their corresponding label or a special label for unknown sequences.
    """
    return sequence_mapping.get(sequence, unknown_label)


def encode_one_hot(train_data, test_data):
    """
    This function applies one-hot encoding to the features in both training and test sets
    """

    # Create a dictionary for sequence mapping in training data
    unique_sequences = np.unique(train_data['sequence'])
    sequence_mapping = {seq: idx for idx, seq in enumerate(unique_sequences)}

    # Define a label for 'Unknown' category
    unknown_label = len(unique_sequences)

    # Apply mapping to training data
    train_data['sequence_mapped'] = train_data['sequence'].apply(
        lambda x: map_sequence(x, sequence_mapping, unknown_label))

    # Apply mapping to test data, mapping unknown sequences to the 'Unknown' category
    test_data['sequence_mapped'] = test_data['sequence'].apply(
        lambda x: map_sequence(x, sequence_mapping, unknown_label))

    # OneHotEncode the mapped sequence data
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.append(train_data[['sequence_mapped']], [[unknown_label]], axis=0))

    # Transform both train and test data
    train_sequence_encoded = encoder.transform(train_data[['sequence_mapped']])
    test_sequence_encoded = encoder.transform(test_data[['sequence_mapped']])

    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(train_sequence_encoded.toarray(), columns=encoder.get_feature_names_out())
    test_encoded_df = pd.DataFrame(test_sequence_encoded.toarray(), columns=encoder.get_feature_names_out())

    # Reset index and concatenate with 'protein' column
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    train_final = pd.concat([train_data['protein'], train_encoded_df], axis=1)
    test_final = pd.concat([test_data['protein'], test_encoded_df], axis=1)

    return train_final, test_final


def generate_kmers(sequence, kmer_size):
    """
    Generate k-mers for a given sequence.
    """

    return [sequence[i:i + kmer_size] for i in range(len(sequence) - kmer_size + 1)]


def apply_kmer(data, kmer_size):
    """
    Apply K-mer encoding to a dataset.
    """

    # Generating k-mers for each sequence
    data['kmer_sequence'] = data['sequence'].apply(lambda x: generate_kmers(x, kmer_size))

    # Flattening the list of k-mers into a single string (required for CountVectorizer)
    data['kmer_sequence'] = data['kmer_sequence'].apply(lambda x: ' '.join(x))

    return data


def encode_kmer(train_data, test_data, kmer_size):
    """
    Use Kmer encoding on the original training and testing data
    """

    # Apply K-mer encoding
    train_data = apply_kmer(train_data, kmer_size)
    test_data = apply_kmer(test_data, kmer_size)

    # Vectorizing using CountVectorizer
    vectorizer = CountVectorizer()

    train_kmer = vectorizer.fit_transform(train_data['kmer_sequence'])
    test_kmer = vectorizer.transform(test_data['kmer_sequence'])

    train_kmer_df = pd.DataFrame(train_kmer.toarray(), columns=vectorizer.get_feature_names_out())
    test_kmer_df = pd.DataFrame(test_kmer.toarray(), columns=vectorizer.get_feature_names_out())

    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_final = pd.concat([train_data['protein'], train_kmer_df], axis=1)
    test_final = pd.concat([test_data['protein'], test_kmer_df], axis=1)

    return train_final, test_final
