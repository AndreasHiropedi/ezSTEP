import dash_bootstrap_components as dbc

from dash import html, dcc, callback, Input, Output, MATCH


def create_layout(model_count):
    """
    This function creates all pages for the model inputs.
    """

    return html.Div(
        id='input-page',
        children=[
            html.Div(
                id='input-page-header',
                children=[
                    html.H1(
                        children=[f"Model {model_count} parameters"]
                    )
                ]
            ),
            html.Div(
                id='input-page-contents',
                children=[
                    model_input_guidelines(),
                    html.Div(
                        id='input-cards-container',
                        children=[
                            dbc.Col(
                                children=[
                                    model_input_parameters_card(model_count)
                                ],
                                style={'width': '40%', 'margin-left': '30px'}
                            ),
                            dbc.Col(
                                children=[
                                    model_input_feature_selection_card(model_count),
                                    model_input_unsupervised_learning_card(model_count),
                                    model_input_hyperparameter_optimisation_card(model_count),
                                ],
                                style={'width': '30%', 'margin-left': '200px'}
                            )
                        ]
                    ),
                    html.Div(
                        id='button-wrapper',
                        children=[
                            html.Button(
                                'Submit model selection',
                                id='input-page-button',
                                n_clicks=0
                            ),
                            html.Button(
                                'Delete model',
                                id='delete-button',
                                n_clicks=0
                            )
                        ]
                    )
                ]
            )
        ]
    )


def model_input_guidelines():
    """
    This function handles the user guidelines for the model inputs page.
    """

    return html.Div(
        id='input-user-guide',
        children=[
            html.H2("User Guidelines"),
            html.P(
                "Below are some guidelines and information about the functionality of the model inputs page,"
                "and some specific information about the individual selection tools available. "
            ),
            html.Div(
                id='input-info-wrapper',
                children=[
                    html.Div(
                        id='input-params-guidelines',
                        children=[
                            html.H4("1. Input parameters"),
                            html.Hr(),
                            html.P(
                                "Here the user selects all the necessary information to create a model (to train and "
                                "test). This includes information such as the model type, the feature descriptor, "
                                "the kmer size (if kmer is selected as a feature descriptor), and the feature "
                                "normalisation algorithm. In addition, the user can opt to enable feature selection, "
                                "unsupervised learning, and hyperparameter optimisation, which would then prompt them "
                                "to input more information (see the sections below)."
                            )
                        ],
                        style={
                            'background': '#f8d7da',
                            'color': '#721c24',
                            'borderColor': '#f5c6cb'
                        }
                    ),
                    html.Div(
                        id='feature-selection-guidelines',
                        children=[
                            html.H4("2. Feature selection"),
                            html.Hr(),
                            html.P(
                                "If the user enables feature selection, they will then need to choose a "
                                "feature selection algorithm, as well as the number of features in the selected model "
                                "to be used by the feature selection algorithm. "
                            )
                        ],
                        style={
                            'background': '#e2e3e5',
                            'color': '#383d41',
                            'borderColor': '#d6d8db'
                        }
                    ),
                    html.Div(
                        id='unsupervised-learning-guidelines',
                        children=[
                            html.H4("3. Unsupervised learning"),
                            html.Hr(),
                            html.P(
                                "If the user enables unsupervised learning, they will then need to choose a "
                                "dimensionality reduction algorithm, as well as the number of dimensions of "
                                "dimensions to be used by the dimensionality reduction algorithm. "
                            )
                        ],
                        style={
                            'background': '#cce5ff',
                            'color': '#004085',
                            'borderColor': '#b8daff'
                        }
                    ),
                    html.Div(
                        id='hyper-opt-guidelines',
                        children=[
                            html.H4("4. Hyperparameter optimisation"),
                            html.Hr(),
                            html.P(
                                "If the user enables hyperparameter optimisation, they will then need to input the "
                                "number of iterations for which they wish to run the Bayesian-Opt hyperparameter "
                                "optimisation algorithm. "
                            )
                        ],
                        style={
                            'background': '#fff3cd',
                            'color': '#856404',
                            'borderColor': '#ffeeba'
                        }
                    )
                ]
            )
        ]
    )


def model_input_parameters_card(model_count):
    """
    This function creates the input parameters card
    """

    return dbc.Card(
        id='parameters-card',
        children=[
            dbc.CardHeader(
                id='card-header-input',
                children=["Input parameters"]
            ),
            dbc.CardBody(
                id='card-body-input',
                children=[
                    model_selection_dropdown(model_count),
                    feature_descriptor_dropdown(model_count),
                    kmer_size_dropdown(model_count),
                    feature_normalization_dropdown(model_count),
                    feature_selection_question(model_count),
                    use_unsupervised_learning_question(model_count),
                    hyperparameter_optimisation_question(model_count)
                ]
            )
        ]
    )


def model_input_feature_selection_card(model_count):
    """
    This function creates the feature selection card
    """

    return dbc.Card(
        id={'type': 'feature-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-input',
                children=['Feature selection']
            ),
            dbc.CardBody(
                id='card-body-input',
                children=[
                    feature_selection_algorithm_dropdown(model_count),
                    feature_number_input(model_count)
                ]
            )
        ],
        style={'display': 'none'}
    )


def model_input_unsupervised_learning_card(model_count):
    """
    This function creates the feature selection card
    """

    return dbc.Card(
        id={'type': 'unsupervised-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-input',
                children=['Unsupervised learning']
            ),
            dbc.CardBody(
                id='card-body-input',
                children=[
                    dimension_reduction_algorithm_dropdown(model_count),
                    dimension_number_input(model_count)
                ]
            )
        ],
        style={'display': 'none'}
    )


def model_input_hyperparameter_optimisation_card(model_count):
    """
    This function creates the hyperparameter optimisation card
    """

    return dbc.Card(
        id={'type': 'hyperparameters-card', 'index': model_count},
        children=[
            dbc.CardHeader(
                id='card-header-input',
                children=['Hyperparameter optimisation']
            ),
            dbc.CardBody(
                id='card-body-input',
                children=[hyperparameter_optimisation_number_input(model_count)]
            )
        ],
        style={'display': 'none'}
    )


def model_selection_dropdown(model_count):
    """
    This function creates the dropdown that allows the user to select
    the type of model they wish to train.
    """

    return html.Div(
        id='model-selection',
        children=[
            html.H6(
                "Select the model type:",
                id='select-model'
            ),
            dcc.Dropdown(
                id={'type': 'model-type-dropdown', 'index': model_count},
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Multi-layer Perceptron', 'value': 'mlp'},
                    {'label': 'Support Vector Machine', 'value': 'svm'},
                    {'label': 'Ridge Regressor', 'value': 'rr'}
                ],
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                },
                searchable=False,
                persistence=True
            )
        ]
    )


def feature_descriptor_dropdown(model_count):
    """
    This function creates the dropdown that allows the user to select
    the type of feature descriptor they wish to use.
    """

    return html.Div(
        id='feature-descriptor',
        children=[
            html.H6(
                "Select feature descriptor:",
                id='select-descriptor'
            ),
            dcc.Dropdown(
                id={'type': 'feature-descriptor-dropdown', 'index': model_count},
                options=[
                    {'label': 'Kmer', 'value': 'kmer'},
                    {'label': 'Binary', 'value': 'binary'},
                ],
                searchable=False,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                },
                persistence=True
            )
        ]
    )


def kmer_size_dropdown(model_count):
    """
    This function creates the dropdown that allows the user to select
    the kmer size to be used (if kmer is selected as the feature descriptor).
    """

    return html.Div(
        id={'type': 'kmer-descriptor', 'index': model_count},
        children=[
            html.H6(
                "Select kmer size:",
                id='select-kmer'
            ),
            dcc.Dropdown(
                id='kmer-size-dropdown',
                options=[
                    {'label': '1', 'value': '1'},
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'},
                    {'label': '4', 'value': '4'},
                    {'label': '5', 'value': '5'},
                ],
                searchable=False,
                persistence=True
            )
        ],
        style={'display': 'none'}
    )


def feature_normalization_dropdown(model_count):
    """
    This function creates the dropdown that allows the user to select
    the type of feature normalization algorithm to be used.
    """

    return html.Div(
        id='normalization-descriptor',
        children=[
            html.H6(
                "Select feature normalization method:",
                id='select-normalization'
            ),
            dcc.Dropdown(
                id={'type': 'feature-normalization-dropdown', 'index': model_count},
                options=[
                    {'label': 'ZScore', 'value': 'zscore'},
                    {'label': 'MinMax', 'value': 'minmax'},
                ],
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                },
                searchable=False,
                persistence=True
            )
        ]
    )


def feature_selection_question(model_count):
    """
    This function allows the user to enable/ disable feature selection
    for the model they have selected.
    """

    return html.Div(
        id='feature-selection-q',
        children=[
            html.H6(
                "Would you like to enable feature selection?",
                id='select-feature'
            ),
            dcc.RadioItems(
                id={'type': 'feature-selection-question', 'index': model_count},
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'},
                ],
                value='no',
                inline=True,
                labelStyle={'margin-right': '50px', 'margin-left': '50px'},
                style={
                    'width': '60%',
                    'font-size': '14pt',
                    'text-align': 'center'
                },
                persistence=True
            )
        ]
    )


def feature_selection_algorithm_dropdown(model_count):
    """
    This function allows the user to choose the feature selection
    algorithm to be used (in case feature selection is enabled).
    """

    return html.Div(
        id='algorithm-descriptor',
        children=[
            html.H6(
                "Select feature selection algorithm:",
                id='select-algorithm'
            ),
            dcc.Dropdown(
                id={'type': 'feature-selection-dropdown', 'index': model_count},
                options=[
                    {'label': 'Chi-Square', 'value': 'chi'},
                    {'label': 'Information Gain', 'value': 'gain'},
                    {'label': 'F-Score', 'value': 'f-score'},
                    {'label': 'Pearson Correlation', 'value': 'pearson'},
                    {'label': 'Mutual Information', 'value': 'mutual'},
                ],
                searchable=False,
                persistence=True,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '20px'
                }
            )
        ]
    )


def feature_number_input(model_count):
    """
    This function allows the user to choose the number of features
    to be used by the feature selection algorithm (in case feature selection
    is enabled).
    """

    return html.Div(
        id='feature-number',
        children=[
            html.H6(
                "Enter number of selected features:",
                id='select-feature-number'
            ),
            dcc.Input(
                id={'type': 'feature-number-input', 'index': model_count},
                type='number',
                min=1,
                max=100,
                step=1,
                persistence=True,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                }
            )
        ]
    )


def use_unsupervised_learning_question(model_count):
    """
    This function allows the user to choose if they wish to use unsupervised
    learning for the visualisations (if yes, the user will need to input some
    dimensionality reduction parameters).
    """

    return html.Div(
        id='unsupervised-learning-q',
        children=[
            html.H6(
                "Would like to use unsupervised learning?",
                id='use-unsupervised'
            ),
            dcc.RadioItems(
                id={'type': 'unsupervised-learning-question', 'index': model_count},
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'},
                ],
                value='no',
                inline=True,
                labelStyle={'margin-right': '50px', 'margin-left': '50px'},
                style={
                    'width': '60%',
                    'font-size': '14pt',
                    'text-align': 'center'
                },
                persistence=True
            )
        ]
    )


def dimension_reduction_algorithm_dropdown(model_count):
    """
    This function allows the user to choose the dimension reduction
    algorithm to be used (in case unsupervised learning is enabled).
    """

    return html.Div(
        id='dimension-algorithm-descriptor',
        children=[
            html.H6(
                "Select dimension reduction method:",
                id='select-dimension-algorithm'
            ),
            dcc.Dropdown(
                id={'type': 'dimension-reduction-dropdown', 'index': model_count},
                options=[
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 'tsne', 'value': 'tsne'},
                ],
                searchable=False,
                persistence=True,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '20px'
                }
            )
        ]
    )


def dimension_number_input(model_count):
    """
    This function allows the user to choose the number of dimensions
    to be used by the dimensionality reduction algorithm (in case unsupervised learning
    is enabled).
    """

    return html.Div(
        id='dimension-number',
        children=[
            html.H6(
                "Enter number of dimensions:",
                id='select-dimension-number'
            ),
            dcc.Input(
                id={'type': 'dimension-number-input', 'index': model_count},
                type='number',
                min=1,
                max=10,
                step=1,
                persistence=True,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                }
            )
        ]
    )


def hyperparameter_optimisation_question(model_count):
    """
    This function allows the user to choose if they wish to use hyperparameter
    optimisation for the selected model.
    """

    return html.Div(
        id='hyper-opt-q',
        children=[
            html.H6(
                f"Would like to use hyperparameter optimisation?",
                id='use-hyper-opt'
            ),
            dcc.RadioItems(
                id={'type': 'hyper-opt-question', 'index': model_count},
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'},
                ],
                value='no',
                inline=True,
                labelStyle={'margin-right': '50px', 'margin-left': '50px'},
                style={
                    'width': '60%',
                    'font-size': '14pt',
                    'text-align': 'center'
                },
                persistence=True
            )
        ]
    )


def hyperparameter_optimisation_number_input(model_count):
    """
    This function allows the user to choose the number of iterations
    to be used by the hyperparameter optimisation algorithm (in case hyperparameter
    optimisation is enabled).
    """

    return html.Div(
        id='loop-number',
        children=[
            html.H6(
                "Enter number of iterations:",
                id='select-iteration-number'
            ),
            dcc.Input(
                id={'type': 'iteration-number-input', 'index': model_count},
                type='number',
                min=1,
                max=100,
                step=1,
                persistence=True,
                style={
                    'width': '91.5%',
                    'font-size': '12pt',
                    'text-align': 'center',
                    'margin-left': '32px'
                }
            )
        ]
    )


@callback(
    Output({'type': 'feature-card', 'index': MATCH}, 'style'),
    [Input({'type': 'feature-selection-question', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')]
)
def enable_feature_selection(answer, model_data):
    """
    This callback function ensures the correct functionality
    for enabling/ disabling feature selection.
    """

    _model_count = model_data['n_clicks']

    if answer == 'yes':
        return {
            'display': 'block',
            'background': 'white',
            'color': 'black',
            'width': '100%',
            'margin-top': '20px',
            'border': '2px solid black',
        }

    elif answer == 'no':
        return {'display': 'none'}


@callback(
    Output({'type': 'unsupervised-card', 'index': MATCH}, 'style'),
    [Input({'type': 'unsupervised-learning-question', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')]
)
def enable_unsupervised(answer, model_data):
    """
    This callback function ensures the correct functionality
    for enabling/ disabling unsupervised learning.
    """

    _model_count = model_data['n_clicks']

    if answer == 'yes':
        return {
            'display': 'block',
            'background': 'white',
            'color': 'black',
            'width': '100%',
            'margin-top': '20px',
            'border': '2px solid black',
        }

    elif answer == 'no':
        return {'display': 'none'}


@callback(
    Output({'type': 'hyperparameters-card', 'index': MATCH}, 'style'),
    [Input({'type': 'hyper-opt-question', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')]
)
def enable_hyperopt(answer, model_data):
    """
    This callback function ensures the correct functionality
    for enabling/ disabling hyperparameter optimisation.
    """

    _model_count = model_data['n_clicks']

    if answer == 'yes':
        return {
            'display': 'block',
            'background': 'white',
            'color': 'black',
            'width': '100%',
            'margin-top': '20px',
            'border': '2px solid black',
        }

    elif answer == 'no':
        return {'display': 'none'}


@callback(
    Output({'type': 'kmer-descriptor', 'index': MATCH}, 'style'),
    [Input({'type': 'feature-descriptor-dropdown', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')]
)
def show_kmer_dropdown(value, model_data):
    """
    This callback function ensures the correct functionality
    for enabling/ disabling Kmer usage for feature description.
    """

    _model_count = model_data['n_clicks']

    if value == 'kmer':
        return {
            'margin-top': '-10px',
            'display': 'flex',
            'align-items': 'center'
        }

    return {'display': 'none'}
