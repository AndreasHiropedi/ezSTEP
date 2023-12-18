import app.globals
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import html, dcc, callback, Input, Output, MATCH
from models.random_forest.random_forest import RandomForest
from models.ridge_regressor.ridge_regressor import RidgeRegressor
from models.mlp.multilayer_perceptron import MultiLayerPerceptron
from models.svm.support_vector_machine import SupportVectorMachine


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

    # Feature selection
    if model.use_feature_select == 'no':
        feature_selection = 'Not enabled'
    else:
        feature_selection = f'{model.feature_number} features selected using {model.feature_selection_algorithm}'

    # Unsupervised learning
    if model.use_unsupervised == 'no':
        unsupervised_learning = 'Not enabled'
    else:
        unsupervised_learning = f'{model.dimensionality_reduction_algorithm} used'

    # Hyperparameter optimization
    if model.use_hyper_opt == 'no':
        hyper_opt = 'Not enabled'
    else:
        hyper_opt = f'Bayesian Hyperparameter Optimization with {model.hyper_opt_iterations} iterations'

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
                                }
                            ),
                            html.P(f"Model Type: {model_type}"),
                            html.P(f"Feature Selection: {feature_selection}"),
                            html.P(f"Unsupervised Learning: {unsupervised_learning}"),
                            html.P(f"Hyperparameter Optimization: {hyper_opt}"),
                        ],
                        style={
                            'flex': 1,
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'space-between',
                            'borderRight': '1px solid black',
                            'paddingRight': '10px',
                        }
                    ),

                    # Training Statistics
                    html.Div(
                        id='card-body-model-training',
                        children=
                        [
                            html.H4(
                                "Training Statistics",
                                style={
                                    'text-align': 'center'
                                }
                            ),
                            html.P(f"RMSE: {round(model.training_RMSE, 2)} ± {round(model.training_RMSE_std, 4)}"),
                            html.P(f"R-squared: {round(model.training_R_squared, 2)} ± "
                                   f"{round(model.training_R_squared_std, 4)}"),
                            html.P(f"MAE: {round(model.training_MAE, 2)} ± {round(model.training_MAE_std, 4)}"),
                            html.P(f"Percentage within 2-fold error: {round(model.training_percentage_2fold_error, 2)} "
                                   f"± {round(model.training_percentage_2fold_error_std, 4)}"),
                        ],
                        style={
                            'flex': 1,
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'space-between',
                            'borderRight': '1px solid black',
                            'paddingRight': '10px',
                            'paddingLeft': '10px',
                        }
                    ),

                    # Testing Statistics
                    html.Div(
                        id='card-body-model-testing',
                        children=[
                            html.H4(
                                "Testing Statistics",
                                style={
                                    'text-align': 'center'
                                }
                            ),
                            html.P(f"RMSE: {round(model.testing_RMSE, 2)}"),
                            html.P(f"R-squared: {round(model.testing_R_squared, 2)}"),
                            html.P(f"MAE: {round(model.testing_MAE, 2)}"),
                            html.P(f"Percentage within 2-fold error: {round(model.testing_percentage_2fold_error, 2)}%")
                        ],
                        style={
                            'flex': 1,
                            'display': 'flex',
                            'flex-direction': 'column',
                            'justify-content': 'space-between',
                            'paddingLeft': '10px',
                        }
                    )
                ],
                style={
                    'display': 'flex'
                }
            )
        ],
        style={
            'width': '75%',
            'margin-left': '200px',
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
    figure.add_trace(go.Scatter(x=actual, y=actual, mode='markers', name='Actual Values'))

    # Add predictions trace
    figure.add_trace(go.Scatter(x=actual, y=predictions, mode='markers', name='Model Predictions'))

    # Customize layout
    figure.update_layout(
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )

    return figure
