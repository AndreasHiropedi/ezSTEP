import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def encode_dna_seq_one_hot(sequence):
    """
    Applies one-hot encoding to an individual DNA sequence
    """

    mapping = dict(zip("acgt", range(4)))
    seq2 = [mapping[i] for i in sequence]
    one_hot_seq = np.eye(4)[seq2]

    return one_hot_seq


def encode_one_hot(train_data, test_data):
    """
    This function applies one-hot encoding to the features in both training and test sets
    """

    one_hot_encoded_train = np.array([encode_dna_seq_one_hot(sequence) for sequence in train_data['sequence']])
    one_hot_encoded_test = np.array([encode_dna_seq_one_hot(sequence) for sequence in test_data['sequence']])

    train_nsamples, train_nx, train_ny = one_hot_encoded_train.shape
    test_nsamples, test_nx, test_ny = one_hot_encoded_test.shape

    train_reshaped = one_hot_encoded_train.reshape((train_nsamples, train_nx * train_ny))
    test_reshaped = one_hot_encoded_test.reshape((test_nsamples, test_nx * test_ny))

    # Create dataframes from reshaped one-hot encoded data
    train_encoded_df = pd.DataFrame(train_reshaped, columns=[f'base_{i}' for i in range(train_nx * train_ny)])
    test_encoded_df = pd.DataFrame(test_reshaped, columns=[f'base_{i}' for i in range(test_nx * test_ny)])

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

    # Ensure kmer_size is an integer
    kmer_size = int(kmer_size)

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
