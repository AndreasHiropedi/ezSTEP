import base64
import dash
import dash_bootstrap_components as dbc
import io
import pandas as pd
import plotly.express as px
import os

from dash import Dash, html, dcc, callback, Input, Output, State, MATCH, dash_table
from flask import Flask
from urllib.parse import urlparse

server = Flask(__name__)
app = Dash(__name__, suppress_callback_exceptions=True, server=server)


def app_header():
    """
    This function builds the header for the web app.
    """

    return html.Header(
        id='app-header',
        children=[
            # Dash logo display
            html.A(
                id='dash-logo',
                children=[
                    html.Img(
                        src='data:image/png;base64,{}'.format(
                            base64.b64encode(
                                open(
                                    './assets/plotly-dash-bio-logo.png', 'rb'
                                ).read()
                            ).decode()
                        ),
                    )
                ],
                href='https://plotly.com',
            ),

            # Web app title
            html.H2(
                "BioNetTrain",
            ),

            # GitHub repo link
            html.A(
                id='github-link',
                children=[
                    "View on GitHub"
                ],
                href='https://github.com/AndreasHiropedi/BioNetTrain',
            ),

            # GitHub logo
            html.Img(
                src='data:image/png;base64,{}'.format(
                    base64.b64encode(
                        open(
                            './assets/GitHub-Mark-Light-64px.png', 'rb'
                        ).read()
                    ).decode()
                ),
            )
        ],
        style={
            'background': '#5F276A',
            'color': 'white'
        }
    )


def app_footer():
    """
    This function builds the footer for the web app.
    """

    return html.Footer(
        id='app-footer',
        children=[
            # University logo
            html.A(
                children=[
                    html.Img(
                        src='data:image/png;base64,{}'.format(
                            base64.b64encode(
                                open(
                                    './assets/eduni-logo.png', 'rb'
                                ).read()
                            ).decode()
                        ),
                    )
                ],
                href='https://homepages.inf.ed.ac.uk/doyarzun/'
            ),

            # Copyright
            html.H3(
                "Biomolecular Control Group ©2024",
            )
        ],
        style={
            'background': 'white',
            'color': 'black'
        }
    )


def user_guide():
    """
    This function builds the user guidelines section of the web app,
    which provides detailed information about how the user can interact
    with the platform.
    """

    return html.Div(
        id='user-guide',
        children=[
            html.H1("User Guidelines"),
            html.P(
                "Below are some guidelines and information about how the platform works. It includes some "
                "general information about the structure and layout of the app, as well as some more "
                "specific information about the individual tools available. "
            ),
            html.Div(
                id='info-wrapper',
                children=[
                    # General information
                    html.Div(
                        id='general-info',
                        children=[
                            html.H4("1. General information"),
                            html.Hr(),
                            html.P(
                                children=[
                                    "The app consists of three sections: ",
                                    html.Br(),
                                    html.Br(),
                                    "   1. File upload",
                                    html.Br(),
                                    html.Br(),
                                    "   2. Model input parameters",
                                    html.Br(),
                                    html.Br(),
                                    "   3. Model output visualisations",
                                    html.Br(),
                                    html.Br(),
                                    "In order for the user to see the model output, their inputted "
                                    "parameters for the selected model, as well as their uploaded "
                                    "dataset, must be first validated and then processed. Once these "
                                    "steps have occurred, the user will be able to visualise the model "
                                    "output (see more in the 'Model output visualisations' section). For "
                                    "more detailed information on each specific subsection, see the "
                                    "information below. "
                                ]
                            )
                        ],
                        style={
                            'background': '#f8d7da',
                            'color': '#721c24',
                            'borderColor': '#f5c6cb'
                        }
                    ),

                    # File upload
                    html.Div(
                        id='file-upload',
                        children=[
                            html.H4("2. File upload"),
                            html.Hr(),
                            html.P(
                                "This section contains three upload boxes, two of which is a required "
                                "field, and one optional field. The required fields are for uploading "
                                "the training and testing data (in order to train the selected model). The "
                                "optional field is for uploading a dataset for querying the model on new "
                                "unseen data that was not part of the training or testing dataset. The user "
                                "will also have the option to choose cross-validation on the training dataset ("
                                "under model inputs) which would override the default setting of using the whole "
                                "dataset uploaded for training and the uploaded test dataset for testing."
                            ),
                            html.P(
                                "For each of the three fields, the user has a choice of how they wish to "
                                "upload the data: they can either upload a file, or paste their data in a "
                                "textbox. If they choose to upload a file, they must ensure the file "
                                "contains at least one column with all the sequence data, and one column "
                                "with all the labels information, with these two columns being matched. If "
                                "they choose to use the textbox, they must ensure the data is formatted in "
                                "the following order: sequence + separator (such as , or | or ;) + label + "
                                "new line character. If the user fails to ensure these conditions, then "
                                "the app may not be able to process their inputted data."
                            )
                        ],
                        style={
                            'background': '#e2e3e5',
                            'color': '#383d41',
                            'borderColor': '#d6d8db'
                        }
                    ),

                    # Model input parameters
                    html.Div(
                        id='model-input',
                        children=[
                            html.H4("3. Model input parameters"),
                            html.Hr(),
                            html.P(
                                "In this section, the user gets to select a model and input all the "
                                "necessary information in order to train and test that model. This "
                                "information will be used together with the datasets uploaded (see the previous "
                                "section for more details) in order to train the models and visualise the "
                                "output (see the 'Model Output Visualisations' section for more). The user "
                                "will also be able to add more than one model, but will need to input all "
                                "the necessary information for each model added. There will also be an  "
                                "option that will allow the user to choose whether or not they wish to  "
                                "optimise the model's hyperparameters or not."
                            ),
                            html.P(
                                "Under this tab, the user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added. In order to input all the necessary "
                                "information, the user will need to click on these hyperlinks individually, "
                                "which will prompt them to a new page where they can input all the data for a "
                                "specific model."
                            )
                        ],
                        style={
                            'background': '#cce5ff',
                            'color': '#004085',
                            'borderColor': '#b8daff'
                        }
                    ),

                    # Model output visualisations
                    html.Div(
                        id='model-output',
                        children=[
                            html.H4("4. Model output visualisations"),
                            html.Hr(),
                            html.P(
                                "Once the data has been uploaded and the user has set all the input "
                                "parameters, the visualisations for the specific model, along with some "
                                "statistics (such as the root mean squared error (RMSE)) are displayed. "
                                "If the user has added several models, then the visualisations for each "
                                "model will be displayed in the model's own container, and a table showing "
                                "all models and their performance statistics will be generated."
                            ),
                            html.P(
                                "Similar to the model inputs, in order to see the visualisations for each "
                                "individual model created, the user will need to click on the appropriate "
                                "hyperlink, which will prompt them to the correct page. However, it is worth "
                                "noting that the table containing all the models and their respective performance "
                                "statistics will be displayed on the main page (note: only the summary statistics "
                                "that have been selected for all models will be displayed in this table, i.e. if "
                                "the user has selected RMSE for model 1 and RMSE and R-squared for model 2, then "
                                "only the RMSE statistic will be displayed in this table; however, the user can still "
                                "see the value of the R-squared statistic on the model's individual output page)."
                            )
                        ],
                        style={
                            'background': '#fff3cd',
                            'color': '#856404',
                            'borderColor': '#ffeeba'
                        }
                    ),
                ]
            )
        ]
    )


def tabs_container():
    """
    This function builds the tab container, which allows the user to switch
    between available tabs.
    """

    return dcc.Tabs(
        id='container',
        value="upload datasets",
        children=[
            dcc.Tab(
                id='tab-upload',
                label="Upload datasets",
                value="upload datasets",
                selected_style={
                    'background': 'grey',
                    'color': 'white'
                }
            ),
            dcc.Tab(
                id='tab-input',
                label="Model input parameters",
                value="model input parameters",
                selected_style={
                    'background': 'grey',
                    'color': 'white'
                }
            ),
            dcc.Tab(
                id='tab-visualise',
                label="Visualise model outputs",
                value="visualise model outputs",
                selected_style={
                    'background': 'grey',
                    'color': 'white'
                }
            )
        ]
    )


def training_data_upload_card(stored_name):
    """
    This function creates the upload boxes for the training data.
    """

    upload_children = None
    success_message = None

    if stored_name:
        upload_children = html.Div(
            [f"Uploaded: {stored_name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {stored_name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Training data (required)"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your training data",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Upload(
                        id='upload-training-data',
                        children=[html.Div([upload_children, success_message])] if stored_name else
                        [
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
                        multiple=False,
                        style={
                            'width': '97.5%',
                            'height': '80px',
                            'textAlign': 'center',
                            'border': '2px dashed black'
                        }
                    ),
                    html.P(
                        "or paste it in the box below",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Textarea(
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        }
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def testing_data_upload_card(stored_name):
    """
    This function creates the upload boxes for the testing data.
    """

    upload_children = None
    success_message = None

    if stored_name:
        upload_children = html.Div(
            [f"Uploaded: {stored_name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {stored_name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Testing data (required)"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your testing data",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Upload(
                        id='upload-testing-data',
                        children=[html.Div([upload_children, success_message])] if stored_name else
                        [
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
                        multiple=False,
                        style={
                            'width': '97.5%',
                            'height': '80px',
                            'textAlign': 'center',
                            'border': '2px dashed black'
                        }
                    ),
                    html.P(
                        "or paste it in the box below",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Textarea(
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        }
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def query_data_upload_card(stored_name):
    """
    This function creates the upload boxes for the model querying data.
    """

    upload_children = None
    success_message = None

    if stored_name:
        upload_children = html.Div(
            [f"Uploaded: {stored_name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {stored_name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Querying data (optional)"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your model querying data",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Upload(
                        id='upload-querying-data',
                        children=[html.Div([upload_children, success_message])] if stored_name else
                        [
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
                        multiple=False,
                        style={
                            'width': '97.5%',
                            'height': '80px',
                            'textAlign': 'center',
                            'border': '2px dashed black'
                        }
                    ),
                    html.P(
                        "or paste it in the box below",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dcc.Textarea(
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        }
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def model_selection_dropdown():
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
                id='model-type-dropdown',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Multi-layer Perceptron', 'value': 'mlp'},
                    {'label': 'Support Vector Machine', 'value': 'svm'},
                    {'label': 'Ridge Regressor', 'value': 'rr'}
                ],
                searchable=False
            )
        ]
    )


def feature_descriptor_dropdown():
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
                id='feature-descriptor-dropdown',
                options=[
                    {'label': 'Kmer', 'value': 'kmer'},
                    {'label': 'Binary', 'value': 'binary'},
                ],
                searchable=False
            )
        ]
    )


def kmer_size_dropdown():
    """
    This function creates the dropdown that allows the user to select
    the kmer size to be used (if kmer is selected as the feature descriptor).
    """

    return html.Div(
        id='kmer-descriptor',
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
                searchable=False
            )
        ]
    )


def feature_normalization_dropdown():
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
                id='feature-normalization-dropdown',
                options=[
                    {'label': 'ZScore', 'value': 'zscore'},
                    {'label': 'MinMax', 'value': 'minmax'},
                ],
                searchable=False
            )
        ]
    )


def feature_selection_question(id_suffix):
    """
    This function allows the user to enable/ disable feature selection
    for the model they have selected.
    """

    return html.Div(
        id=f'feature-selection-q-{id_suffix}',
        children=[
            html.H6(
                "Would you like to enable feature selection?",
                id='select-feature'
            ),
            dcc.RadioItems(
                id=f'feature-selection-question-{id_suffix}',
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
                }
            )
        ],
        style={
            'margin-top': '-40px',
            'display': 'flex',
            'align-items': 'center'
        }
    )


def output_statistics_dropdown():
    """
    This function allows the user to select the types of summary statistics
    they wish to see regarding the chosen model (such as RMSE, R-squared, etc.).
    """

    return html.Div(
        id='output-statistics',
        children=[
            html.H6(
                "Select the evaluation statistics about the model that you wish to see in the output (select all that "
                "apply):",
                id='select-statistics'
            ),
            dcc.Dropdown(
                id='output-statistics-dropdown',
                options=[
                    {'label': 'RMSE', 'value': 'rmse'},
                    {'label': 'R-squared', 'value': 'r-squared'},
                ],
                multi=True,
                searchable=False
            )
        ]
    )


def feature_selection_algorithm_dropdown():
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
                id='feature-selection-dropdown',
                options=[
                    {'label': 'Chi-Square', 'value': 'chi'},
                    {'label': 'Information Gain', 'value': 'gain'},
                    {'label': 'F-Score', 'value': 'f-score'},
                    {'label': 'Pearson Correlation', 'value': 'pearson'},
                    {'label': 'Mutual Information', 'value': 'mutual'},
                ],
                searchable=False
            )
        ]
    )


def feature_number_input():
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
                id='feature-number-input',
                type='number',
                min=1,
                max=100,
                step=1
            )
        ]
    )


def use_unsupervised_learning_question(id_suffix):
    """
    This function allows the user to choose if they wish to use unsupervised
    learning for the visualisations (if yes, the user will need to input some
    dimensionality reduction parameters).
    """

    return html.Div(
        id=f'unsupervised-learning-{id_suffix}',
        children=[
            html.H6(
                f"Would like to use unsupervised learning?",
                id='use-unsupervised'
            ),
            dcc.RadioItems(
                id=f'unsupervised-learning-question-{id_suffix}',
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
                }
            )
        ],
        style={
            'margin-top': '-40px',
            'display': 'flex',
            'align-items': 'center'
        }
    )


def dimension_reduction_algorithm_dropdown():
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
                id='dimension-reduction-dropdown',
                options=[
                    {'label': 'PCA', 'value': 'pca'},
                    {'label': 'tsne', 'value': 'tsne'},
                ],
                searchable=False
            )
        ]
    )


def dimension_number_input():
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
                id='dimension-number-input',
                type='number',
                min=1,
                max=10,
                step=1
            )
        ]
    )


def use_cross_validation_question(id_suffix):
    """
    This function allows the user to choose if they wish to use cross-validation
    on the training dataset, hence overriding the default setting.
    """

    return html.Div(
        id=f'cross-validation-{id_suffix}',
        children=[
            html.H6(
                f"Would like to perform cross-validation on the training dataset?",
                id='use-cross-validation'
            ),
            dcc.RadioItems(
                id=f'cross-validation-question-{id_suffix}',
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
                }
            )
        ],
        style={
            'margin-top': '-40px',
            'display': 'flex',
            'align-items': 'center'
        }
    )


def hyperparameter_optimisation_question(id_suffix):
    """
    This function allows the user to choose if they wish to use hyperparameter
    optimisation for the selected model.
    """

    return html.Div(
        id=f'hyper-opt-{id_suffix}',
        children=[
            html.H6(
                f"Would like to use hyperparameter optimisation?",
                id='use-hyper-opt'
            ),
            dcc.RadioItems(
                id=f'hyper-opt-question-{id_suffix}',
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
                }
            )
        ],
        style={
            'margin-top': '-40px',
            'display': 'flex',
            'align-items': 'center'
        }
    )


def hyperparameter_optimisation_number_input():
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
                id='iteration-number-input',
                type='number',
                min=1,
                max=100,
                step=1
            )
        ]
    )


def feature_selection_section():
    """
    This function creates the feature selection section for the model inputs.
    """

    return html.Div(
        id='feature-selection',
        children=[
            html.H3('Feature selection', id='input-header'),
            feature_selection_algorithm_dropdown(),
            feature_number_input()
        ]
    )


def dimension_reduction_section():
    """
    This function creates the dimension reduction section for the model inputs.
    """

    return html.Div(
        id='dimension-reduction',
        children=[
            html.H3('Dimension reduction', id='input-header'),
            dimension_reduction_algorithm_dropdown(),
            dimension_number_input()
        ]
    )


def hyperparameter_optimisation_section():
    """
    This function creates the hyperparameter optimisation section for the model inputs.
    """

    return html.Div(
        id='hyperparameter-optimisation',
        children=[
            html.H3('Hyperparameter optimisation', id='input-header'),
            hyperparameter_optimisation_number_input()
        ]
    )


def model_input_parameters_card(model_count):
    """
    This function creates all information cards for the model inputs.
    """

    return dbc.Card(
        id='card-input',
        children=[
            dbc.CardHeader(
                id='card-header-input',
                children=[f"Model {model_count} parameters"]
            ),
            dbc.CardBody(
                id='card-body-input',
                children=[
                    html.Div(
                        id='input-parameters',
                        children=[
                            html.H3('Input parameters', id='input-header'),
                            model_selection_dropdown(),
                            feature_descriptor_dropdown(),
                            kmer_size_dropdown(),
                            feature_normalization_dropdown(),
                            feature_selection_question(str(model_count)),
                            use_unsupervised_learning_question(str(model_count)),
                            use_cross_validation_question(str(model_count)),
                            hyperparameter_optimisation_question(str(model_count))
                        ],
                    ),
                    html.Hr(),
                    html.Div(
                        id='output-features',
                        children=[
                            html.H3('Output statistics', id='input-header'),
                            output_statistics_dropdown()
                        ]
                    ),
                    html.Div(
                        id={'type': 'feature-selection-section', 'index': str(model_count)},
                        children=[
                            html.Hr(),
                            feature_selection_section()
                        ],
                        style={'display': 'block'} # change this to none
                    ),
                    html.Div(
                        id={'type': 'dimension-reduction-section', 'index': str(model_count)},
                        children=[
                            html.Hr(),
                            dimension_reduction_section()
                        ],
                        style={'display': 'block'} # change this to none
                    ),
                    html.Div(
                        id={'type': 'hyper-opt-section', 'index': str(model_count)},
                        children=[
                            html.Hr(),
                            hyperparameter_optimisation_section()
                        ],
                        style={'display': 'block'} # change this to none
                    ),
                ]
            )
        ]
    )


def model_output_card(model_count):
    """
    This function creates all information cards for the model outputs.
    """

    return dbc.Card(
        id='card-output',
        children=[
            dbc.CardHeader(
                id='card-header-output',
                children=[f"Model {model_count} output"]
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    # Placeholder
                    html.H4('Interactive scatter plot with Iris dataset'),
                    dcc.Graph(id={'type': 'scatter-plot', 'index': model_count}),
                    html.P("Filter by petal width:"),
                    dcc.RangeSlider(
                        id={'type': 'range-slider', 'index': model_count},
                        min=0, max=2.5, step=0.1,
                        marks={0: '0', 2.5: '2.5'},
                        value=[0.5, 2]
                    )
                ]
            )
        ]
    )


def model_input_ref(model_count):
    """
    This function creates hyperlinks to a separate page for each
    model input (for each model created).
    """

    return dcc.Link(
        children=[
            html.H4(
                f"Model {model_count} input parameters",
                id='model-inputs-ref'
            )
        ],
        href=f'/model-input/model-{model_count}'
    )


def model_output_ref(model_count):
    """
    This function creates hyperlinks to a separate page for each
    model output (for each model created).
    """

    return dcc.Link(
        children=[
            html.H4(
                f"Model {model_count} output",
                id='model-outputs-ref'
            )
        ],
        href=f'/model-output/model-{model_count}'
    )


def model_output_performance_statistics_table(model_count):
    """
    This function displays the output summary statistics table
    for the models selected by the user (contains only the summary statistics
    selected for all models).
    """

    # Create a dataframe of all the model output statistics
    df = pd.DataFrame({
        'Model number': [f'Model {i}' for i in range(1, int(model_count)+1)],
        'RMSE': [f'Model {i} RMSE' for i in range(1, int(model_count)+1)],
        'R-squared': [f'Model {i} R-squared' for i in range(1, int(model_count)+1)]
    })

    return dash_table.DataTable(
        id='table',
        columns=[{"Model number": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_data={
            'text-align': 'center',
            'font-size': '12pt',
            'font-family': 'Times',
            'border': '1px solid black'
        },
        style_table={
            'width': '80%',
            'margin-top': '40px',
            'margin-left': '180px'
        },
        style_header={
            'text-align': 'center',
            'font-size': '12pt',
            'font-family': 'Times',
            'font-weight': 'bold',
            'background': '#5F276A',
            'color': 'white',
            'border': '1px solid black'
        }
    )


@app.callback(
    [Output('upload-training-data', 'children'),
     Output('store-uploaded-train-file', 'data')],
    Input('upload-training-data', 'contents'),
    State('upload-training-data', 'filename')
)
def update_training_output(content, name):
    if content:
        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        failed_message = html.Div(
            [f"File {name} not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Process the content if you need, e.g., for a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@app.callback(
    [Output('upload-testing-data', 'children'),
     Output('store-uploaded-test-file', 'data')],
    Input('upload-testing-data', 'contents'),
    State('upload-testing-data', 'filename')
)
def update_testing_output(content, name):
    if content:
        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        failed_message = html.Div(
            [f"File {name} not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Process the content if you need, e.g., for a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@app.callback(
    [Output('upload-querying-data', 'children'),
     Output('store-uploaded-query-file', 'data')],
    Input('upload-querying-data', 'contents'),
    State('upload-querying-data', 'filename')
)
def update_querying_output(content, name):
    if content:
        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {name}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {name} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        failed_message = html.Div(
            [f"File {name} not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Process the content if you need, e.g., for a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@app.callback(
    Output({'type': 'feature-selection-section', 'index': MATCH}, 'style'),
    Input({'type': 'feature-selection-question', 'index': MATCH}, 'value')
)
def update_feature_selection(value):
    """
    This callback function displays (or doesn't) the feature selection
    section on the model input card based on user's input.
    """

    if value == 'yes':
        return {'display': 'block'}

    elif value == 'no':
        return {'display': 'none'}


@callback(
    Output({'type': 'dimension-reduction-section', 'index': MATCH}, 'style'),
    Input({'type': 'unsupervised-learning-question', 'index': MATCH}, 'value')
)
def update_unsupervised_learning(value):
    """
    This callback function displays (or doesn't) the dimension reduction
    section on the model input card based on user's input.
    """

    if value == 'yes':
        return {'display': 'block'}

    elif value == 'no':
        return {'display': 'none'}


@callback(
    Output({'type': 'scatter-plot', 'index': MATCH}, 'figure'),
    [Input({'type': 'range-slider', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')])
def update_bar_chart(slider_range, model_data):
    """
    This callback function keeps track of the user changes to the
    displayed graph (as part of the output information cards)
    """

    model_count = model_data['n_clicks']

    df = px.data.iris()  # replace with your own data source
    low, high = slider_range
    mask = (df['petal_width'] > low) & (df['petal_width'] < high)
    fig = px.scatter(
        df[mask], x="sepal_width", y="sepal_length",
        color="species", size='petal_length',
        hover_data=['petal_width'])

    return fig


@callback(
    Output('content', 'children'),
    Input('container', 'value'),
    [State('store-model-content', 'data'),
     State('store-model-count', 'data'),
     State('store-uploaded-train-file', 'data'),
     State('store-uploaded-test-file', 'data'),
     State('store-uploaded-query-file', 'data')
     ]
)
def render_tabs_content(
    selected_tab,
    stored_content,
    stored_count,
    stored_train_file,
    stored_test_file,
    stored_query_file
):
    """
    This callback function keeps track of the user changes to the
    tabs container (and displays the correct information for each tab)
    """

    # File upload tab
    if selected_tab == 'upload datasets':
        training_name = stored_train_file['filename'] if stored_train_file else None
        testing_name = stored_test_file['filename'] if stored_test_file else None
        querying_name = stored_query_file['filename'] if stored_query_file else None

        return dbc.Container(
            id='tabs-content-upload',
            children=[
                dbc.Row(
                    id='card-row',
                    children=[
                        dbc.Col(
                            children=[training_data_upload_card(training_name)],
                            md=3,
                            style={
                                'margin-left': '50px',
                                'margin-right': '50px'
                            }
                        ),
                        dbc.Col(
                            children=[testing_data_upload_card(testing_name)],
                            md=5,
                            style={
                                'margin-left': '50px',
                                'margin-right': '50px'
                            }
                        ),
                        dbc.Col(
                            children=[query_data_upload_card(querying_name)],
                            md=3,
                            style={
                                'margin-left': '80px',
                                'margin-right': '80px'
                            }
                        )
                    ],
                    justify="center"
                )
            ],
            style={
                'background': 'white',
                'color': 'black'
            },
            fluid=True
        )

    # Model inputs tab
    elif selected_tab == 'model input parameters':
        # If we switch tabs, this restores the previous state (so that all models created are preserved)
        if stored_content:
            return dbc.Row(
                id='tabs-content-input',
                children=stored_content['children']
            )
        # Initial state
        else:
            return dbc.Row(
                id='tabs-content-input',
                children=[
                    # This creates the initial layout with one model
                    model_input_ref("1"),
                    html.Button(
                        'Add a new model',
                        id='button',
                        n_clicks=1
                    )
                ]
            )

    # Model outputs
    elif selected_tab == "visualise model outputs":
        return dbc.Row(
            id='tabs-content-output',
            children=[model_output_performance_statistics_table(stored_count['n_clicks'])] +
            [model_output_ref(i) for i in range(1, stored_count['n_clicks']+1)]
        )

    # Validation check
    else:
        return 'No content available.'


@app.callback(
    [Output('tabs-content-input', 'children'),
     Output('store-model-count', 'data'),
     Output('store-model-content', 'data')
     ],
    Input('button', 'n_clicks'),
    [State('tabs-content-input', 'children'),
     State('store-model-count', 'data'),
     State('store-model-content', 'data')
     ]
)
def add_new_model(n_clicks, current_children, stored_count, stored_content):
    """
    This callback function keeps track of the user changes to the
    model inputs tab (when adding new models).
    """

    # Check if a new model has been added
    if n_clicks > stored_count['n_clicks']:
        stored_count['n_clicks'] = n_clicks
        children = current_children + [model_input_ref(n_clicks)]
        new_content = {
            'children': children
        }
        return children, stored_count, new_content

    # If there has been no new model added
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'href'),
    State('store-model-count', 'data'),
    State('store-model-content', 'data'),
    State('store-uploaded-train-file', 'data'),
    State('store-uploaded-test-file', 'data'),
    State('store-uploaded-query-file', 'data')
)
def display_page(href, stored_count, stored_content, stored_train_file, stored_test_file,  stored_query_file):
    """
    This callback allows for switching between tabs when choosing to view
    individual model inputs/ outputs.
    """

    # Extract pathname from the full URL (href)
    parsed_url = urlparse(href)
    pathname = parsed_url.path

    # Instead of always setting the initial data, check if data exists
    initial_count_data = {'n_clicks': 1} if not stored_count else stored_count
    initial_content_data = {} if not stored_content else stored_content
    initial_train_file = {} if not stored_train_file else stored_train_file
    initial_test_file = {} if not stored_test_file else stored_test_file
    initial_query_file = {} if not stored_query_file else stored_query_file

    if pathname.startswith('/model-input/'):
        # If a model inputs tab is selected, return the card for that input
        try:
            model_num = int(pathname.split('/')[-1][-1])
            return model_input_parameters_card(model_num)
        except ValueError:
            return html.Div('Invalid model number.')

    elif pathname.startswith('/model-output/'):
        # If a model output tab is selected, return the card for that output
        try:
            model_num = int(pathname.split('/')[-1][-1])
            return model_output_card(model_num)
        except ValueError:
            return html.Div('Invalid model number.')

    return [
        app_header(),
        html.Div(
            id='app-contents',
            children=[
                user_guide(),
                html.Div(
                    id='tabs-container',
                    children=[
                        tabs_container(),
                        html.Div(id='content'),
                        dcc.Store(id='store-model-count', data=initial_count_data),
                        dcc.Store(id='store-model-content', data=initial_content_data),
                        dcc.Store(id='store-uploaded-train-file', data=initial_train_file),
                        dcc.Store(id='store-uploaded-test-file', data=initial_test_file),
                        dcc.Store(id='store-uploaded-query-file', data=initial_query_file),
                        app_footer()
                    ]
                )
            ]
        )
    ]


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        id='page-content',
        children=[
            app_header(),
            html.Div(
                id='app-contents',
                children=[
                    user_guide(),
                    html.Div(
                        id='tabs-container',
                        children=[
                            tabs_container(),
                            html.Div(id='content'),
                            dcc.Store(id='store-model-count', data={'n_clicks': 1}, storage_type='session'),
                            dcc.Store(id='store-model-content', data={}, storage_type='session'),
                            dcc.Store(id='store-uploaded-train-file', storage_type='session'),
                            dcc.Store(id='store-uploaded-test-file', storage_type='session'),
                            dcc.Store(id='store-uploaded-query-file', storage_type='session'),
                            app_footer()
                        ]
                    )
                ]
            )
        ]
    )
])

if __name__ == '__main__':
    app.run_server(port=int(os.environ.get("PORT", 8050)))
