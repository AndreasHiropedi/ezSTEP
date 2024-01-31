import pandas as pd

import app.globals
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import umap

from dash import html, dcc, callback, Input, Output, MATCH, dash_table
from models.random_forest.random_forest import RandomForest
from models.ridge_regressor.ridge_regressor import RidgeRegressor
from models.mlp.multilayer_perceptron import MultiLayerPerceptron
from models.svm.support_vector_machine import SupportVectorMachine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_layout(model_count):
    """
    This function creates all pages for the model outputs.
    """

    # Check if model has been created successfully
    model_key = f'Model {model_count}'

    if app.globals.MODELS_LIST[model_key] is None:
        return html.Div(
            id='output-page',
            children=[
                html.Div(
                    id='output-page-header',
                    children=[
                        html.H1(
                            children=[f"Model {model_count} output"]
                        )
                    ]
                ),
                html.Div(
                    id='output-page-contents',
                    children=[
                        "No content available since the model hasn't been created or initiated."
                    ]
                )
            ]
        )

    return html.Div(
        id='output-page',
        children=[
            html.Div(
                id='output-page-header',
                children=[
                    html.H1(
                        children=[f"Model {model_count} output"]
                    )
                ]
            ),
            html.Div(
                id='output-page-contents',
                children=[
                    model_summary_card(model_count),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[output_statistics_card(model_count)],
                                md=4
                            ),
                            dbc.Col(
                                children=[predicted_versus_actual_card(model_count)],
                                md=3
                            )
                        ],
                        justify="center",
                        style={
                            'display': 'flex',
                            'width': '100%',
                        }
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[training_feature_correlation_card(model_count)],
                                md=4
                            ),
                            dbc.Col(
                                children=[testing_feature_correlation_card(model_count)],
                                md=3
                            )
                        ],
                        justify="center",
                        style={
                            'display': 'flex',
                            'width': '100%',
                        }
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[querying_feature_correlation_card(model_count)],
                                md=4
                            ),
                            dbc.Col(
                                children=[querying_file_download_card(model_count)],
                                md=3
                            )
                        ],
                        justify="center",
                        style={
                            'display': 'flex',
                            'width': '100%',
                        }
                    ),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                children=[explained_variance_plot_card(model_count)],
                                md=4
                            ),
                            dbc.Col(
                                children=[unsupervised_learning_plot_card(model_count)],
                                md=3
                            )
                        ],
                        justify="center",
                        style={
                            'display': 'flex',
                            'width': '100%',
                            'margin-bottom': '50px'
                        }
                    )
                ]
            )
        ]
    )


def model_summary_card(model_count):
    """
    This function generates the card containing the summary information about the respective model
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    # Retrieve all necessary summary information

    # Model type
    model_type = ''
    if isinstance(model, RandomForest):
        model_type = 'Random Forest'
    elif isinstance(model, MultiLayerPerceptron):
        model_type = 'Multi-layer Perceptron'
    elif isinstance(model, SupportVectorMachine):
        model_type = 'Support Vector Machine'
    elif isinstance(model, RidgeRegressor):
        model_type = 'Ridge Regressor'

    # Feature encoding
    feature_encoding = ''
    if model.feature_encoding_method == 'binary':
        feature_encoding = 'One-hot'
    elif model.feature_encoding_method == 'kmer':
        feature_encoding = f'Kmer ({model.kmer_size})'

    # Feature normalisation
    feature_normalisation = ''
    if model.feature_normalization_algorithm == 'minmax':
        feature_normalisation = 'MinMax'
    elif model.feature_normalization_algorithm == 'zscore':
        feature_normalisation = 'Z-score'

    # Feature selection
    if model.use_feature_select == 'no':
        feature_selection = 'Not enabled'
    else:
        feature_selection = f'Top {model.feature_number} features selected using {model.feature_selection_algorithm}'

    # Unsupervised learning
    if model.use_unsupervised == 'no':
        unsupervised_learning = 'Not enabled'
    else:
        unsupervised_learning = f'{model.dimensionality_reduction_algorithm} selected'

    # Hyperparameter optimization
    if model.use_hyper_opt == 'no':
        hyper_opt = 'Not enabled'
    else:
        hyper_opt = f'Bayesian Hyperparameter Optimization with {model.hyper_opt_iterations} iterations'

    # Training output statistics
    training_rmse = f"{round(model.training_RMSE, 2)} ± {round(model.training_RMSE_std, 4)}"
    training_r_squared = f"{round(model.training_R_squared, 2)} ± {round(model.training_R_squared_std, 4)}"
    training_mae = f"{round(model.training_MAE, 2)} ± {round(model.training_MAE_std, 4)}"
    training_2fold = f"{round(model.training_percentage_2fold_error, 2)} ± " \
                     f"{round(model.training_percentage_2fold_error_std, 4)}"

    # Testing output statistics
    testing_rmse = f"{round(model.testing_RMSE, 2)}"
    testing_r_squared = f"{round(model.testing_R_squared, 2)}"
    testing_mae = f"{round(model.testing_MAE, 2)}"
    testing_2fold = f"{round(model.testing_percentage_2fold_error, 2)}"

    # Dataframe that stores the model summary information
    model_summary_data = pd.DataFrame(
        {
            'Information type': ['Model type', 'Feature encoding', 'Feature normalisation',
                                 'Feature selection', 'Unsupervised learning',
                                 'Hyperparameter optimization'],
            'Details': [model_type, feature_encoding, feature_normalisation, feature_selection,
                        unsupervised_learning, hyper_opt]
        }
    )

    # Dataframe that stores the model's output statistics
    model_output_statistics = pd.DataFrame(
        {
           'Statistic name': ['RMSE', 'R-squared', 'MAE', 'Percentage within 2-fold error (%)'],
           'Training value': [training_rmse, training_r_squared, training_mae, training_2fold],
           'Testing value': [testing_rmse, testing_r_squared, testing_mae, testing_2fold]
        }
    )

    return dbc.Card(
        id={'type': 'model-summary-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-model',
                children=['Summary information']
            ),
            dbc.CardBody(
                id='card-body-model',
                children=[

                    # Model Input Details
                    html.Div(
                        id='card-body-model-info',
                        children=
                        [
                            html.H4(
                                "Model Input Details",
                                style={
                                    'text-align': 'center',
                                    'font-size': '14pt'
                                }
                            ),
                            html.Div(
                                id='model-input-details-table',
                                children=[
                                    dash_table.DataTable(
                                        id='summary-table',
                                        columns=[{"name": i, "id": i} for i in model_summary_data.columns],
                                        data=model_summary_data.to_dict('records'),
                                        style_cell={
                                            'text-align': 'center',
                                            'font-size': '12pt',
                                            'font-family': 'Times',
                                            'border': '1px solid black'
                                        },
                                        style_header={
                                            'font-weight': 'bold',
                                            'background': 'blue',
                                            'color': 'white',
                                        },
                                        editable=False,
                                        cell_selectable=False
                                    )
                                ],
                                style={
                                    'margin-bottom': '40px'
                                }
                            )
                        ]
                    ),
                    html.Hr(
                        style={
                            'margin-left': '-15px',
                            'margin-right': '-15px',
                            'height': '2px',
                            'color': 'black',
                            'background': 'black'
                        }
                    ),

                    # Output Statistics
                    html.Div(
                        id='card-body-model-statistics',
                        children=[
                            html.H4(
                                "Model training and testing output statistics",
                                style={
                                    'text-align': 'center',
                                    'font-size': '14pt'
                                }
                            ),
                            html.Div(
                                id='summary-statistics-table',
                                children=[
                                    dash_table.DataTable(
                                        id='statistics-table',
                                        columns=[{"name": i, "id": i} for i in model_output_statistics.columns],
                                        data=model_output_statistics.to_dict('records'),
                                        style_cell={
                                            'text-align': 'center',
                                            'font-size': '12pt',
                                            'font-family': 'Times',
                                            'border': '1px solid black'
                                        },
                                        style_header={
                                            'font-weight': 'bold',
                                            'background': 'orange',
                                            'color': 'black',
                                        },
                                        editable=False,
                                        cell_selectable=False
                                    )
                                ],
                                style={
                                    'margin-bottom': '40px'
                                }
                            )
                        ]
                    )
                ]
            )
        ],
        style={
            'width': '83.5%',
            'margin-left': '120px',
            'border': '2px solid black',
            'margin-top': '50px'
        }
    )


def output_statistics_card(model_count):
    """
    This function generates the card containing the spider plot of training versus
    testing output statistics.
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    return dbc.Card(
        id={'type': 'train-test-statistics-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-stats',
                children=['Training vs Testing statistics']
            ),
            dbc.CardBody(
                id='card-body-stats',
                children=[
                    dcc.Graph(
                        figure=output_statistics_graph(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '120px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px'
        }
    )


def output_statistics_graph(model):
    """
    This function generates the spider plot for training versus testing
    output statistics.
    """

    figure = go.Figure()

    categories = ['RMSE', 'R-squared', 'MAE', 'Two-fold error', 'RMSE']

    # Add training statistics trace
    figure.add_trace(go.Scatterpolar(
        r=[model.training_RMSE, model.training_R_squared, model.training_MAE,
           model.training_2fold_error, model.training_RMSE],
        theta=categories,
        name='Training statistics'
    ))

    # Add testing statistics trace
    figure.add_trace(go.Scatterpolar(
        r=[model.testing_RMSE, model.testing_R_squared, model.testing_MAE,
            model.testing_2fold_error, model.testing_RMSE],
        theta=categories,
        name='Testing statistics'
    ))

    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )),
        showlegend=True
    )

    return figure


def predicted_versus_actual_card(model_count):
    """
    This function generates the card containing the scatter plot of predicted versus
    actual output.
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    return dbc.Card(
        id={'type': 'predict-versus-actual-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-predict',
                children=['Actual versus Predicted plot']
            ),
            dbc.CardBody(
                id='card-body-predict',
                children=[
                    dcc.Graph(
                        figure=predicted_versus_actual_graph(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '160px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px'
        }
    )


def predicted_versus_actual_graph(model):
    """
    This function generates the scatter plot of predicted versus
    actual output.
    """

    # Data
    actual = model.testing_data['protein']
    predictions = model.model_predictions

    # Create scatter plot
    figure = go.Figure()

    # Add actual data trace
    figure.add_trace(go.Scatter(x=actual, y=actual, mode='lines', name='Actual Values'))

    # Add predictions trace
    figure.add_trace(go.Scatter(x=actual, y=predictions, mode='markers', name='Model Predictions'))

    # Customize layout
    figure.update_layout(
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )

    return figure


def training_feature_correlation_card(model_count):
    """
    This function generates the card containing the plot for the feature correlation
    to the target variable for the training data
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    return dbc.Card(
        id={'type': 'train-feature-correlation-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-train-feature',
                children=['Training data feature correlation with target variable']
            ),
            dbc.CardBody(
                id='card-body-train-feature',
                children=[
                    dcc.Graph(
                        figure=training_data_feature_correlation_plot(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '120px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px'
        }
    )


def training_data_feature_correlation_plot(model):
    """
    This function generates the bar chart showing feature correlation
    to the target variable for the training data.
    """

    # Get the data
    data = model.encoded_train

    # Identify feature columns (all columns except the target)
    features = data.columns.tolist()
    features.remove('protein')

    # Calculate the correlation with the target variable
    correlation_with_target = data[features].corrwith(data['protein'])

    # Separate positive and negative correlations
    positive_features = [feature for feature, corr in correlation_with_target.items() if corr > 0]
    positive_correlations = [corr for corr in correlation_with_target if corr > 0]

    negative_features = [feature for feature, corr in correlation_with_target.items() if corr <= 0]
    negative_correlations = [corr for corr in correlation_with_target if corr <= 0]

    # Create the bar plot with two traces, one for positive and one for negative correlations
    figure = go.Figure()

    # Trace for positive correlations
    figure.add_trace(
        go.Bar(
            y=positive_correlations,
            x=positive_features,
            name='Positive Correlation',
            marker=dict(color='blue'),
            orientation='v'
        )
    )

    # Trace for negative correlations
    figure.add_trace(
        go.Bar(
            y=negative_correlations,
            x=negative_features,
            name='Negative Correlation',
            marker=dict(color='red'),
            orientation='v'
        )
    )

    # Update layout for a better look
    figure.update_layout(
        xaxis_title='Features',
        yaxis_title='Correlation Coefficient',
    )

    return figure


def testing_feature_correlation_card(model_count):
    """
    This function generates the card containing the plot for the feature correlation
    to the target variable for the test data.
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    return dbc.Card(
        id={'type': 'test-feature-correlation-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-test-feature',
                children=['Test data feature correlation with target variable']
            ),
            dbc.CardBody(
                id='card-body-test-feature',
                children=[
                    dcc.Graph(
                        figure=testing_data_feature_correlation_plot(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '160px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px'
        }
    )


def testing_data_feature_correlation_plot(model):
    """
    This function generates the bar chart showing feature correlation
    to the target variable for the test data.
    """

    # Get the data
    data = model.encoded_test

    # Identify feature columns (all columns except the target)
    features = data.columns.tolist()
    features.remove('protein')

    # Calculate the correlation with the target variable
    correlation_with_target = data[features].corrwith(data['protein'])

    # Separate positive and negative correlations
    positive_features = [feature for feature, corr in correlation_with_target.items() if corr > 0]
    positive_correlations = [corr for corr in correlation_with_target if corr > 0]

    negative_features = [feature for feature, corr in correlation_with_target.items() if corr <= 0]
    negative_correlations = [corr for corr in correlation_with_target if corr <= 0]

    # Create the bar plot with two traces, one for positive and one for negative correlations
    figure = go.Figure()

    # Trace for positive correlations
    figure.add_trace(
        go.Bar(
            y=positive_correlations,
            x=positive_features,
            name='Positive Correlation',
            marker=dict(color='blue'),
            orientation='v'
        )
    )

    # Trace for negative correlations
    figure.add_trace(
        go.Bar(
            y=negative_correlations,
            x=negative_features,
            name='Negative Correlation',
            marker=dict(color='red'),
            orientation='v'
        )
    )

    # Update layout for a better look
    figure.update_layout(
        xaxis_title='Features',
        yaxis_title='Correlation Coefficient',
    )

    return figure


def querying_feature_correlation_card(model_count):
    """
    This function generates the card containing the plot for the feature correlation
    to the target variable for the query data (displayed only if the query dataset
    is provided).
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    if model.querying_data is None:
        return html.Div(
            style={
                'max-height': '0px',
                'max-width': '0px'
            }
        )

    return dbc.Card(
        id={'type': 'query-feature-correlation-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-query-feature',
                children=['Query data feature correlation with target variable']
            ),
            dbc.CardBody(
                id='card-body-query-feature',
                children=[
                    dcc.Graph(
                        figure=querying_data_feature_correlation_plot(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '120px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px'
        }
    )


def querying_data_feature_correlation_plot(model):
    """
    This function generates the bar chart showing feature correlation
    to the target variable for the query data (displayed only if the
    query dataset is provided).
    """

    # Get the data
    data = model.encoded_query

    # Identify feature columns (all columns except the target)
    features = data.columns.tolist()

    # Calculate the correlation with the target variable
    correlation_with_target = data[features].corrwith(model.query_predictions['protein'])

    # Separate positive and negative correlations
    positive_features = [feature for feature, corr in correlation_with_target.items() if corr > 0]
    positive_correlations = [corr for corr in correlation_with_target if corr > 0]

    negative_features = [feature for feature, corr in correlation_with_target.items() if corr <= 0]
    negative_correlations = [corr for corr in correlation_with_target if corr <= 0]

    # Create the bar plot with two traces, one for positive and one for negative correlations
    figure = go.Figure()

    # Trace for positive correlations
    figure.add_trace(
        go.Bar(
            y=positive_correlations,
            x=positive_features,
            name='Positive Correlation',
            marker=dict(color='blue'),
            orientation='v'
        )
    )

    # Trace for negative correlations
    figure.add_trace(
        go.Bar(
            y=negative_correlations,
            x=negative_features,
            name='Negative Correlation',
            marker=dict(color='red'),
            orientation='v'
        )
    )

    # Update layout for a better look
    figure.update_layout(
        xaxis_title='Features',
        yaxis_title='Correlation Coefficient',
    )

    return figure


def querying_file_download_card(model_count):
    """
    This function generates the card containing the downloadable CSV file
    containing the model's predictions for the uploaded querying data
    (displayed only if the query dataset is provided).
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    # Generate file name
    file_name = f"{model.model_number}_query_predictions.csv"

    if model.querying_data is None:
        return html.Div(
            style={
                'max-height': '0px',
                'max-width': '0px'
            }
        )

    return dbc.Card(
        id={'type': 'query-download-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-query-predictions',
                children=['Query model predictions']
            ),
            dbc.CardBody(
                id='card-body-query-predictions',
                children=[
                    html.P(
                        "The CSV file containing the model's predictions for the provided query dataset can be "
                        "downloaded from the link below: "
                    ),
                    html.Div(
                        children=[
                            dbc.Button(
                                html.H4(
                                    f"Download {file_name}",
                                    style={
                                        'font-size': '12pt',
                                        'margin-top': '5px'
                                    }
                                ),
                                id={'type': "btn-download", 'index': model_count},
                                n_clicks=0,
                                style={
                                    'border': '2px solid black',
                                    'cursor': 'pointer',
                                    'height': '40px',
                                    'background': 'purple',
                                    'color': 'white',
                                }
                            ),
                            dcc.Download(
                                id={'type': "download-link", 'index': model_count}
                            )
                        ],
                        style={
                            'margin-top': '100px',
                            'text-size': '12pt',
                            'margin-left': '100px',
                            'margin-bottom': '100px',
                        }
                    ),
                    html.P(
                        "Note that these predictions are unique for this model, and every model will generate "
                        "its own unique set of predictions despite using the same query data.",
                        style={
                            'margin-top': '0px'
                        }
                    ),
                ]
            )
        ],
        style={
            'margin-left': '160px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px',
            'height': '498.5px'
        }
    )


def explained_variance_plot_card(model_count):
    """
    This function generates the card containing the PCA plot for the explained variance
    for the selected features using feature selection (displayed only if feature selection
    is enabled).
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    if model.use_feature_select == 'no':
        return html.Div(
            style={
                'max-height': '0px',
                'max-width': '0px'
            }
        )

    selected_features = model.feature_number
    feature_selection_method = model.feature_selection_algorithm

    return dbc.Card(
        id={'type': 'explained-variance-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-explained-variance',
                children=[f'Explained variance for the top {selected_features} features selected using '
                          f'{feature_selection_method}']
            ),
            dbc.CardBody(
                id='card-body-explained-variance',
                children=[
                    dcc.Graph(
                        figure=explained_variance_plot(model)
                    )
                ]
            )
        ],
        style={
            'margin-left': '120px',
            'margin-right': '40px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px',
            'height': '100%'
        }
    )


def explained_variance_plot(model):
    """
    This function generates the PCA plot for the explained variance for the selected features
    chosen by the selected method for feature selection (displayed only if feature selection
    is enabled).
    """

    # Data
    data = model.selected_train

    # Apply PCA
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)

    # Get the explained and cumulative variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = explained_var.cumsum()

    # Create the figure
    figure = go.Figure()

    # Adding individual explained variance (bar chart)
    figure.add_trace(go.Bar(
        x=[f'PC{i + 1}' for i in range(data.shape[1])],
        y=explained_var,
        name='Individual Explained Variance'
    ))

    # Adding cumulative explained variance (line chart)
    figure.add_trace(go.Scatter(
        x=[f'PC{i + 1}' for i in range(data.shape[1])],
        y=cumulative_var,
        name='Cumulative Explained Variance'
    ))

    # Updating layout
    figure.update_layout(
        xaxis_title='Principal Components',
        yaxis_title='Explained Variance Ratio',
    )

    return figure


def unsupervised_learning_plot_card(model_count):
    """
    This function generates the card containing the unsupervised learning plot based
    on the unsupervised learning method selected (displayed only if unsupervised learning
    is enabled).
    """

    # Retrieve the model
    model_key = f'Model {model_count}'
    model = app.globals.MODELS_LIST[model_key]

    if model.use_unsupervised == 'no':
        return html.Div(
            style={
                'max-height': '0px',
                'max-width': '0px'
            }
        )

    unsupervised_learning_method = model.dimensionality_reduction_algorithm

    # Display the correct graph based on the chosen unsupervised learning method
    graph_container = None
    if unsupervised_learning_method == 'PCA':
        graph_container = html.Div(
            id='pca-plot-container',
            children=[
                dcc.Graph(
                    id={'type': 'pca-plot', 'index': model_count},
                    figure=initial_pca_plot(model)
                ),
                html.Div(
                    children=[
                        html.Label(
                            'SVD Solver:',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.Dropdown(
                            id={'type': 'svd-solver-dropdown', 'index': model_count},
                            options=[
                                {'label': 'Auto', 'value': 'auto'},
                                {'label': 'Full', 'value': 'full'},
                                {'label': 'Arpack', 'value': 'arpack'},
                                {'label': 'Randomized', 'value': 'randomized'}
                            ],
                            value='auto',
                            searchable=False,
                            persistence=True,
                            style={
                                'width': '90%',
                                'font-size': '12pt',
                                'text-align': 'center',
                                'margin-left': '20px'
                            }
                        ),
                    ],
                    style={
                        'margin-top': '20px',
                        'margin-bottom': '20px',
                        'display': 'flex',
                        'align-items': 'center'
                    }
                ),
                html.Div(
                    children=[
                        html.Label(
                            'Whiten:',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.RadioItems(
                            id={'type': 'whiten-radio', 'index': model_count},
                            options=[
                                {'label': 'True', 'value': True},
                                {'label': 'False', 'value': False}
                            ],
                            value=False,
                            persistence=True,
                            inline=True,
                            labelStyle={'margin-right': '50px', 'margin-left': '50px'},
                            style={
                                'width': '60%',
                                'font-size': '12pt',
                                'text-align': 'center'
                            },
                        )
                    ],
                    style={
                        'margin-top': '20px',
                        'display': 'flex',
                        'align-items': 'center'
                    }
                )
            ]
        )

    elif unsupervised_learning_method == 't-SNE':
        graph_container = html.Div(
            id='tsne-plot-container',
            children=[
                dcc.Graph(
                    id={'type': 'tsne-plot', 'index': model_count},
                    figure=initial_tsne_plot(model)
                ),
                html.Div(
                    children=[
                        html.Label(
                            'Perplexity:',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.Slider(
                            id={'type': 'perplexity-slider', 'index': model_count},
                            min=5,
                            max=50,
                            step=5,
                            value=30,
                            persistence=True,
                        )
                    ],
                    style={
                        'margin-top': '20px',
                        'margin-bottom': '20px'
                    }
                ),
                html.Div(
                    children=[
                        html.Label(
                            'Learning Rate:',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.Input(
                            id={'type': 'learning-rate-input', 'index': model_count},
                            type='number',
                            value=200,
                            min=10,
                            max=1000,
                            persistence=True,
                            style={
                                'width': '70%',
                                'font-size': '12pt',
                                'text-align': 'center',
                                'margin-left': '32px'
                            }
                        )
                    ],
                    style={
                        'margin-top': '20px',
                        'display': 'flex',
                        'align-items': 'center'
                    }
                )
            ]
        )

    elif unsupervised_learning_method == 'UMAP':
        graph_container = html.Div(
            id='umap-plot-container',
            children=[
                dcc.Graph(
                    id={'type': 'umap-plot', 'index': model_count},
                    figure=initial_umap_plot(model)
                ),
                html.Div(
                    children=[
                        html.Label(
                            'n_neighbors',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.Slider(
                            id={'type': 'n-neighbors-slider', 'index': model_count},
                            min=2,
                            max=50,
                            value=15,
                            marks={i: str(i) for i in range(2, 51, 5)},
                            step=1,
                            persistence=True,
                        )
                    ],
                    style={
                        'margin-top': '20px',
                        'margin-bottom': '20px'
                    }
                ),
                html.Div(
                    children=[
                        html.Label(
                            'min_dist',
                            style={
                                'margin-left': '20px'
                            }
                        ),
                        dcc.Slider(
                            id={'type': 'min-dist-slider', 'index': model_count},
                            min=0.0,
                            max=1.0,
                            value=0.0,
                            marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                            step=0.1,
                            persistence=True,
                        )
                    ],
                    style={
                        'margin-top': '20px'
                    }
                )
            ]
        )

    return dbc.Card(
        id={'type': 'unsupervised-plot-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-unsupervised-plot',
                children=[f"{unsupervised_learning_method} 2D plot"] if unsupervised_learning_method != 'UMAP' else
                [f"{unsupervised_learning_method} 2D plot of the training data"]
            ),
            dbc.CardBody(
                id='card-body-unsupervised-plot',
                children=[
                    dcc.Loading(
                        children=[graph_container] if unsupervised_learning_method != 't-SNE' else
                                 [
                                     graph_container,
                                     html.P(
                                         "NOTE: the allowed range of values is 10 to 1000",
                                         style={
                                             'margin-top': '10px',
                                             'font-size': '10pt',
                                             'text-align': 'center',
                                             'margin-bottom': '-20px'
                                         }
                                     )
                                 ]
                    )
                ]
            )
        ],
        style={
            'margin-left': '120px',
            'border': '2px solid black',
            'margin-top': '50px',
            'width': '550px',
            'height': '100%'
        }
    )


def initial_pca_plot(model):
    """
    This function generates the initial PCA plot based on the standard parameter values
    """

    # Data
    components_train = model.unsupervised_train
    components_test = model.unsupervised_test
    components_query = model.unsupervised_query

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=components_train[:, 0],
            y=components_train[:, 1],
            mode='markers',
            name='Training Data'
        )
    )

    figure.add_trace(
        go.Scatter(
            x=components_test[:, 0],
            y=components_test[:, 1],
            mode='markers',
            name='Test Data'
        )
    )

    if components_query is not None:
        figure.add_trace(
            go.Scatter(
                x=components_query[:, 0],
                y=components_query[:, 1],
                mode='markers',
                name='Query Data'
            )
        )

    # Update the figure layout
    figure.update_layout(
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component"
    )

    return figure


def initial_tsne_plot(model):
    """
    This function generates the initial t-SNE plot based on the standard parameter values
    """

    # Data
    tsne_results_train = model.unsupervised_train
    tsne_results_test = model.unsupervised_test
    tsne_results_query = model.unsupervised_query

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=tsne_results_train[:, 0],
            y=tsne_results_train[:, 1],
            mode='markers',
            name='Training Data'
        )
    )

    figure.add_trace(
        go.Scatter(
            x=tsne_results_test[:, 0],
            y=tsne_results_test[:, 1],
            mode='markers',
            name='Test Data'
        )
    )

    if tsne_results_query is not None:
        figure.add_trace(
            go.Scatter(
                x=tsne_results_query[:, 0],
                y=tsne_results_query[:, 1],
                mode='markers',
                name='Query Data'
            )
        )

    # Update the figure layout
    figure.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2'
    )

    return figure


def initial_umap_plot(model):
    """
    This function generates the initial UMAP plot based on the standard parameter values
    """

    # Data
    umap_results_train = model.unsupervised_train
    labels = model.normalized_train['protein']

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=umap_results_train[:, 0],
            y=umap_results_train[:, 1],
            mode='markers',
            marker=dict(color=labels, colorscale='Viridis', showscale=True)
        )
    )

    # Update the figure layout
    figure.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2'
    )

    return figure


@callback(
    Output({'type': "download-link", 'index': MATCH}, "data"),
    Input({'type': "btn-download", 'index': MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def generate_csv(_n_clicks):
    """
    This callback generates the CSV file with the model's predictions
    for the query dataset (if one is provided) and allows the user to
    download this CSV.
    """

    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update

    # Extract the index part from the triggered_id
    triggered_id = ctx.triggered[0]['prop_id']
    new_split = triggered_id.split(".")
    big_string = new_split[0].strip("{}")
    another_split = big_string.split(",")

    # Get index value from index part
    index_part = another_split[0]
    index_value = index_part[-1]

    model = app.globals.MODELS_LIST[f'Model {index_value}']

    # Data
    df = model.model_query_created_file

    # File name to be used
    file_name = f"{model.model_number}_query_data_predictions.csv"

    # Send the CSV string
    return dcc.send_data_frame(df.to_csv, file_name, index=False)


@callback(
    Output({'type': 'pca-plot', 'index': MATCH}, 'figure'),
    [Input({'type': 'svd-solver-dropdown', 'index': MATCH}, 'value'),
     Input({'type': 'whiten-radio', 'index': MATCH}, 'value')],
    prevent_initial_call=True
)
def update_pca_graph(svd_solver, whiten):
    """
    This callback generates the interactive PCA plot (if unsupervised learning is enabled)
    """

    ctx = dash.callback_context

    # Extract the index part from the triggered_id
    triggered_id = ctx.triggered[0]['prop_id']
    new_split = triggered_id.split(".")
    big_string = new_split[0].strip("{}")
    another_split = big_string.split(",")

    # Get index value from index part
    index_part = another_split[0]
    index_value = index_part[-1]

    model = app.globals.MODELS_LIST[f'Model {index_value}']

    # Get the data and initially fitted PCA outputs
    data_train = model.normalized_train
    data_test = model.normalized_test
    data_query = model.normalized_query

    # Variable to keep track of query data (if provided)
    components_query = None

    # Refit PCA
    pca = PCA(n_components=2, svd_solver=svd_solver, whiten=whiten)
    components_train = pca.fit_transform(data_train)
    components_test = pca.transform(data_test)
    if data_query is not None:
        components_query = pca.transform(data_query)

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=components_train[:, 0],
            y=components_train[:, 1],
            mode='markers',
            name='Training Data'
        )
    )

    figure.add_trace(
        go.Scatter(
            x=components_test[:, 0],
            y=components_test[:, 1],
            mode='markers',
            name='Test Data'
        )
    )

    if components_query is not None:
        figure.add_trace(
            go.Scatter(
                x=components_query[:, 0],
                y=components_query[:, 1],
                mode='markers',
                name='Query Data'
            )
        )

    # Update the figure layout
    figure.update_layout(
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component"
    )

    return figure


@callback(
    Output({'type': 'tsne-plot', 'index': MATCH}, 'figure'),
    [Input({'type': 'perplexity-slider', 'index': MATCH}, 'value'),
     Input({'type': 'learning-rate-input', 'index': MATCH}, 'value')],
    prevent_initial_call=True
)
def update_tsne_plot(perplexity, learning_rate):
    """
    This callback generates the interactive t-SNE plot (if unsupervised learning is enabled)
    """

    ctx = dash.callback_context

    # Extract the index part from the triggered_id
    triggered_id = ctx.triggered[0]['prop_id']
    new_split = triggered_id.split(".")
    big_string = new_split[0].strip("{}")
    another_split = big_string.split(",")

    # Get index value from index part
    index_part = another_split[0]
    index_value = index_part[-1]

    model = app.globals.MODELS_LIST[f'Model {index_value}']

    # Variable to keep track of query data (if provided)
    tsne_results_query = None

    # Data
    data_train = model.normalized_train
    data_test = model.normalized_test
    data_query = model.normalized_query

    # Refit the t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, init='pca')
    tsne_results_train = tsne.fit_transform(data_train)
    tsne_results_test = tsne.fit_transform(data_test)
    if data_query is not None:
        tsne_results_query = tsne.fit_transform(data_query)

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=tsne_results_train[:, 0],
            y=tsne_results_train[:, 1],
            mode='markers',
            name='Training Data'
        )
    )

    figure.add_trace(
        go.Scatter(
            x=tsne_results_test[:, 0],
            y=tsne_results_test[:, 1],
            mode='markers',
            name='Test Data'
        )
    )

    if tsne_results_query is not None:
        figure.add_trace(
            go.Scatter(
                x=tsne_results_query[:, 0],
                y=tsne_results_query[:, 1],
                mode='markers',
                name='Query Data'
            )
        )

    # Update the figure layout
    figure.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2'
    )

    return figure


@callback(
    Output({'type': 'umap-plot', 'index': MATCH}, 'figure'),
    [Input({'type': 'n-neighbors-slider', 'index': MATCH}, 'value'),
     Input({'type': 'min-dist-slider', 'index': MATCH}, 'value')],
    prevent_initial_call=True
)
def update_umap_plot(n_neighbors, min_dist):
    """
    This callback generates the interactive UMAP plot (if unsupervised learning is enabled)
    """

    ctx = dash.callback_context

    # Extract the index part from the triggered_id
    triggered_id = ctx.triggered[0]['prop_id']
    new_split = triggered_id.split(".")
    big_string = new_split[0].strip("{}")
    another_split = big_string.split(",")

    # Get index value from index part
    index_part = another_split[0]
    index_value = index_part[-1]

    model = app.globals.MODELS_LIST[f'Model {index_value}']

    # Data
    data_train = model.normalized_train
    labels = model.normalized_train['protein']

    # Re-fit the UMAP
    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_results_train = umap_model.fit_transform(data_train)

    # Create the figure
    figure = go.Figure()

    # Add data to the figure
    figure.add_trace(
        go.Scatter(
            x=umap_results_train[:, 0],
            y=umap_results_train[:, 1],
            mode='markers',
            marker=dict(color=labels, colorscale='Viridis', showscale=True)
        )
    )

    # Update the figure layout
    figure.update_layout(
        xaxis_title='Component 1',
        yaxis_title='Component 2'
    )

    return figure
