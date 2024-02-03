import umap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def use_umap(train_data, test_data, query_data):
    """
    This function implements the UMAP dimensionality reduction algorithm.
    """

    # Separate features from target
    train_features = train_data.drop('protein', axis=1)
    test_features = test_data.drop('protein', axis=1)

    # Apply UMAP
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0, random_state=42, low_memory=True)
    train_umap = umap_model.fit_transform(train_features)
    test_umap = umap_model.transform(test_features)

    # Transform query set (if applicable)
    if query_data is not None:
        query_umap = umap_model.transform(query_data)

        return train_umap, test_umap, query_umap

    return train_umap, test_umap, None


def use_pca(train_data, test_data, query_data):
    """
    This function implements the PCA dimensionality reduction algorithm.
    """

    # Separate features from target
    train_features = train_data.drop('protein', axis=1)
    test_features = test_data.drop('protein', axis=1)

    # Apply PCA
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)

    # Transform query set (if applicable)
    if query_data is not None:
        query_pca = pca.transform(query_data)

        return train_pca, test_pca, query_pca

    return train_pca, test_pca, None


def use_tsne(train_data, test_data, query_data):
    """
    This function implements the t-sne dimensionality reduction algorithm.
    """

    # Separate features from target
    train_features = train_data.drop('protein', axis=1)
    test_features = test_data.drop('protein', axis=1)

    # Apply t-SNE with PCA initialization
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    train_tsne = tsne.fit_transform(train_features)
    test_tsne = tsne.fit_transform(test_features)

    # Transform query set (if applicable)
    if query_data is not None:
        query_tsne = tsne.fit_transform(query_data)

        return train_tsne, test_tsne, query_tsne

    return train_tsne, test_tsne, None
