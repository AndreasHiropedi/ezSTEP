import base64
import dash
import io
import os
import time
import uuid

import dash_bootstrap_components as dbc
import pandas as pd

from . import globals
from . import model_inputs_page
from . import model_outputs_page
from . import output_statistics_page

from dash import html, dcc, callback, Input, Output, State, clientside_callback, Dash
from flask import send_from_directory, session
from flask_apscheduler import APScheduler
from urllib.parse import urlparse


# Basic definitions for the app
my_app = Dash(__name__)
server = my_app.server
my_app.config.suppress_callback_exceptions = True

# Set a secret key for the user session
server.secret_key = 'your_secret_key'

# Initialize the scheduler with the Flask server
scheduler = APScheduler()
scheduler.init_app(server)
scheduler.start()


def app_header():
    """
    This function builds the header for the web app.
    """

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the image file
    plotly_image_path = os.path.join(current_directory, 'assets', 'plotly-dash-bio-logo.png')
    # Open the image, read it, and encode it into Base64
    encoded_plotly_image = base64.b64encode(open(plotly_image_path, 'rb').read()).decode()

    # Construct the absolute path to the image file
    logo_image_path = os.path.join(current_directory, 'assets', 'app-logo.png')
    # Open the image, read it, and encode it into Base64
    encoded_logo_image = base64.b64encode(open(logo_image_path, 'rb').read()).decode()

    # Construct the absolute path to the image file
    github_image_path = os.path.join(current_directory, 'assets', 'GitHub-Mark-Light-64px.png')
    # Open the image, read it, and encode it into Base64
    encoded_github_image = base64.b64encode(open(github_image_path, 'rb').read()).decode()

    return html.Header(
        id='app-header',
        children=[
            # Dash logo display
            html.A(
                id='dash-logo',
                children=[
                    html.Img(
                        src=f'data:image/png;base64,{encoded_plotly_image}'
                    )
                ],
                href='https://plotly.com',
                target='_blank'
            ),

            # App logo
            html.A(
                id='app-logo',
                children=[
                    html.Img(
                        src=f'data:image/png;base64,{encoded_logo_image}'
                    ),
                ],
                href='/'
            ),

            # GitHub repo link
            html.A(
                id='github-link',
                children=[
                    "View on GitHub"
                ],
                href='https://github.com/AndreasHiropedi/ezSTEP',
                target='_blank'
            ),

            # GitHub logo
            html.Img(
                src=f'data:image/png;base64,{encoded_github_image}'
            )
        ],
        style={
            'background': 'black',
            'color': 'white'
        }
    )


def app_footer():
    """
    This function builds the footer for the web app.
    """

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the image file
    uni_image_path = os.path.join(current_directory, 'assets', 'eduni-logo.png')
    # Open the image, read it, and encode it into Base64
    encoded_uni_image = base64.b64encode(open(uni_image_path, 'rb').read()).decode()

    return html.Footer(
        id='app-footer',
        children=[
            # University logo
            html.A(
                children=[
                    html.Img(
                        src=f'data:image/png;base64,{encoded_uni_image}'
                    )
                ],
                href='https://homepages.inf.ed.ac.uk/doyarzun/',
                target='_blank'
            ),

            # Copyright
            html.H3(
                "Biomolecular Control Group 2024",
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

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the image file
    upload_image_path = os.path.join(current_directory, 'assets', 'upload.png')
    # Open the image, read it, and encode it into Base64
    encoded_upload_image = base64.b64encode(open(upload_image_path, 'rb').read()).decode()

    # Construct the absolute path to the image file
    input_image_path = os.path.join(current_directory, 'assets', 'inputs.png')
    # Open the image, read it, and encode it into Base64
    encoded_input_image = base64.b64encode(open(input_image_path, 'rb').read()).decode()

    # Construct the absolute path to the image file
    output_image_path = os.path.join(current_directory, 'assets', 'outputs.png')
    # Open the image, read it, and encode it into Base64
    encoded_output_image = base64.b64encode(open(output_image_path, 'rb').read()).decode()

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
                                    "   3. Model outputs",
                                    html.Br(),
                                    html.Br(),
                                    "In order for the user to see the model output, their inputted "
                                    "parameters for the selected model, as well as their uploaded "
                                    "dataset, must be first validated and then processed. Once these "
                                    "steps have occurred, the user will be able to visualise the model "
                                    "output (see more in the 'Model outputs' section). For "
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
                            html.Img(
                                src=f'data:image/png;base64,{encoded_upload_image}'
                            ),
                            html.P(
                                "As can be seen in the image above, this section contains three upload boxes. "
                                "The required fields are for uploading "
                                "the training and testing data (in order to train the selected model), and the "
                                "optional field is for uploading a dataset for querying the model on unlabelled data. "
                                "For each of the three fields, the user has a choice of how they wish to "
                                "upload the data: they can either upload a file, or paste their data in a "
                                "textbox. If they choose to upload a file, they must ensure the file "
                                "contains at least one column with all the sequence data, and one column "
                                "with all the labels information, with these two columns being matched. If "
                                "they choose to use the textbox, they must ensure the data is formatted in "
                                "the following order: sequence + separator (such as , or | or ;) + label + "
                                "new line character. If the user fails to ensure these conditions, then "
                                "the app will not be able to process their inputted data and will inform the user. "
                                "However, it is worth noting "
                                "that these conditions only apply for training and testing, since for the querying "
                                "data the user will only need to provide the sequences (in the format of one sequence "
                                "per line)."
                            ),
                            html.P(
                                "NOTE: for the training process, we always perform 5-fold cross validation on the "
                                "uploaded training dataset. Therefore, it is worth noting that, if the length of the "
                                "text input is less than 5 sequences long, even if the sequences are valid, the input "
                                "will be rendered invalid. We provide a set of example files below, one for each of "
                                "the three steps (training, testing, querying), which can be downloaded and used for "
                                "running the app's machine learning pipeline."
                            ),
                            html.Div(
                                children=[
                                    html.A(
                                        'example_train_data.csv',
                                        download='example_train_data.csv',
                                        href='/downloadable_data/example_train_data.csv',
                                        style={
                                            'margin-left': '220px'
                                        }
                                    ),
                                    html.A(
                                        'example_test_data.csv',
                                        download='example_test_data.csv',
                                        href='/downloadable_data/example_test_data.csv',
                                        style={
                                            'margin-left': '220px'
                                        }
                                    ),
                                    html.A(
                                        'example_query_data.csv',
                                        download='example_query_data.csv',
                                        href='/downloadable_data/example_query_data.csv',
                                        style={
                                            'margin-left': '220px'
                                        }
                                    ),
                                ],
                                style={
                                    'margin-bottom': '15px'
                                }
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
                            html.Img(
                                src=f'data:image/png;base64,{encoded_input_image}'
                            ),
                            html.P(
                                "In this section, the user gets to select a model and input all the "
                                "necessary information in order to train and test that model. This "
                                "information will be used together with the datasets uploaded (see the previous "
                                "section for more details) in order to train the models and visualise the "
                                "output (see the 'Model Outputs' section for more). The user "
                                "will also be able to add more than one model, but will need to input all "
                                "the necessary information for each model added. There will also be an  "
                                "option that will allow the user to choose whether or not they wish to  "
                                "optimise the model's hyperparameters or not."
                            ),
                            html.P(
                                "The user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added (see image). "
                                "In order to input all the necessary "
                                "information, the user will need to click on these hyperlinks individually, "
                                "which will prompt them to a new page where they can input all the data for a "
                                "specific model (see image). More information about the specifics of "
                                "model inputs can be found "
                                "in the user guidelines on the individual model input pages."
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
                            html.H4("4. Model outputs"),
                            html.Hr(),
                            html.Img(
                                src=f'data:image/png;base64,{encoded_output_image}'
                            ),
                            html.P(
                                children=[
                                    "Once the data has been uploaded and the user has set all the input parameters, "
                                    "the visualisations for the specific model, along with a spider plot showing "
                                    "several output statistics, are generated. These statistics are calculated for "
                                    "each model, and these are all displayed in the spider plot for each model "
                                    "(see image). "
                                    "The four main summary statistics used are: "
                                    "Root Mean Squared Error (RMSE), "
                                    "R-squared, "
                                    "Mean Absolute Error (MAE), "
                                    "and Percentage (%) within 2-fold error."
                                ]
                            ),
                            html.P(
                                "Similarly to model inputs, the user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added (see image). "
                                "On each of these pages, there will be several "
                                "graphs displayed (such as a scatter plot of predicted versus actual values, or bar "
                                "charts showing the feature correlation to the target variable (protein)), along with "
                                "a summary of the user's selected inputs and the model's performance in training and "
                                "testing. Depending on the additional inputs the user provides (such as query data, or "
                                "enabling feature selection), additional graphs will be generated and displayed on "
                                "these individual pages."
                                "All plots available in this app are interactive, "
                                "and using the legend on the side, the user can select "
                                "which models they wish to have displayed in the graph, and they can simply "
                                "enable/ disable them by clicking on the respective item in the legend. Additionally, "
                                "all graphs provide features such as zooming in/ out, saving the graph as a PNG to "
                                "the user's local device, and selecting to focus only on certain regions of the graph."
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
                id='tab-output',
                label="Model outputs",
                value="model outputs",
                selected_style={
                    'background': 'grey',
                    'color': 'white'
                }
            )
        ]
    )


def training_data_upload_card():
    """
    This function creates the upload boxes for the training data.
    """

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
                    html.Div(
                        id='upload-container',
                        children=[
                            dcc.Upload(
                                id='upload-training-data',
                                children=[
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
                            )
                        ]
                    ),
                    html.P(
                        "or paste it in the box below",
                        style={
                            'text-align': 'center',
                            'font-size': '14pt'
                        }
                    ),
                    dbc.Textarea(
                        id='text-train-data',
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        },
                        persistence=True,
                        persistence_type='session'
                    ),
                    dcc.Input(
                        id='previous-train-value',
                        type='hidden',
                        value='',
                        persistence=True,
                        persistence_type='session'
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def testing_data_upload_card():
    """
    This function creates the upload boxes for the testing data.
    """

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
                        children=[
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
                    dbc.Textarea(
                        id='text-test-data',
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        },
                        persistence=True,
                        persistence_type='session'
                    ),
                    dcc.Input(
                        id='previous-test-value',
                        type='hidden',
                        value='',
                        persistence=True,
                        persistence_type='session'
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def query_data_upload_card():
    """
    This function creates the upload boxes for the model querying data.
    """

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
                        children=[
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
                        id='text-query-data',
                        style={
                            'width': '97.5%',
                            'height': '100px'
                        },
                        persistence=True,
                        persistence_type='session'
                    ),
                    dcc.Input(
                        id='previous-query-value',
                        type='hidden',
                        value='',
                        persistence=True,
                        persistence_type='session'
                    )
                ]
            )
        ],
        className="mx-auto"
    )


def output_statistics_card():
    """
    This function creates the card with the output statistics dropdown
    """

    return dbc.Card(
        id='output-statistics-card',
        children=[
            dbc.CardHeader(
                id='card-header-output',
                children=['Testing output statistics']
            ),
            dbc.CardBody(
                id='card-body-output',
                children=[
                    html.P(
                        "Click the hyperlink below to view the plot of all testing output statistics "
                        "for all existing models."
                    ),
                    output_metric_ref()
                ]
            )
        ]
    )


def model_input_ref(model_key, session_id):
    """
    This function creates hyperlinks to a separate page for each
    model input (for each model created).
    """

    # Get user data
    user_data = globals.get_user_session_data(session_id)
    models_list = user_data['MODELS_LIST']

    if model_key not in models_list.keys():
        models_list[model_key] = None
        user_data['MODELS_LIST'] = models_list
        globals.store_user_session_data(session_id, user_data)

    return html.A(
        children=[
            html.H4(
                f"{model_key} input parameters",
                id='model-inputs-ref'
            )
        ],
        href=f'/model-input/{model_key}',
        target='_blank'
    )


def output_metric_ref():
    """
    This function creates the hyperlink to a separate page for the
    output statistics graph
    """

    return html.A(
        children=[
            html.H4(
                "View output statistics plot",
                id='metric-plot-ref'
            )
        ],
        href=f'/output-statistics/output-graph',
        target='_blank'
    )


def model_output_ref(model_key):
    """
    This function creates hyperlinks to a separate page for each
    model output (for each model created).
    """

    return html.A(
        children=[
            html.H4(
                f"{model_key} output",
                id='model-outputs-ref'
            )
        ],
        href=f'/model-output/{model_key}',
        target='_blank'
    )


@callback(
    [Output('upload-training-data', 'children'),
     Output('store-uploaded-train-file', 'data')],
    Input('upload-training-data', 'contents'),
    [State('upload-training-data', 'filename'),
     State('store-uploaded-train-file', 'data'),
     State('session-id', 'data')]
)
def update_training_output(content, name, stored_train_file_name, session_data):
    """
    This callback updates the contents of the upload box for the
    model train data.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    if stored_train_file_name and not content:
        file = stored_train_file_name['filename']

        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {file}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {file} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        final_display = html.Div([upload_children, success_message])
        return final_display, dash.no_update

    elif content:
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

        wrong_format_message = html.Div(
            [f"File {name} is not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        wrong_columns_message = html.Div(
            [f"File {name} use the wrong column names!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        invalid_data_message = html.Div(
            [f"File {name} uses data in the wrong format!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        limit_exceeded_message = html.Div(
            [f"File {name} is too large or sequences are too long!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Check if file is a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Check if file contains two columns labeled 'sequence' and 'protein'
            df.columns = df.columns.astype(str).str.lower()
            if 'sequence' in df.columns and 'protein' in df.columns:

                # Check data in the protein column is of the correct type
                parsed_protein = pd.to_numeric(df['protein'], errors='coerce')
                if df["protein"].dtype != 'float64' or not parsed_protein.notna().all():
                    final_display = html.Div([upload_children, invalid_data_message])
                    return final_display, None

                # Check data in the sequence column is of the correct type
                # we check that the data is a string, all sequences are the same length
                # and all sequences contain only a mix of the characters A, C, G, and T and nothing else
                invalid_sequences = df['sequence'].str.lower().str.contains('[^actg]')
                if df['sequence'].dtype != object or df['sequence'].str.len().nunique() != 1 or invalid_sequences.any():
                    final_display = html.Div([upload_children, invalid_data_message])
                    return final_display, None

                # For memory's sake, limit size file and the length of each sequence
                if len(df) > 20000 or df['sequence'].str.len()[0] > 250:
                    final_display = html.Div([upload_children, limit_exceeded_message])
                    return final_display, None

                df['sequence'] = df['sequence'].str.lower()
                final_display = html.Div([upload_children, success_message])

                # Set the training file name and data in REDIS for that user
                user_data['TRAINING_FILE'] = name
                user_data['TRAINING_DATA'] = df.to_json()
                globals.store_user_session_data(session_id, user_data)

                return final_display, {'filename': name}

            # If CSV but without right columns
            final_display = html.Div([upload_children, wrong_columns_message])
            return final_display, None

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, wrong_format_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-train-data', 'style'),
    Output('training-change-counter', 'data'),
    Output('previous-train-value', 'value'),
    Input('text-train-data', 'value'),
    State('previous-train-value', 'value'),
    State('store-uploaded-train-file', 'data'),
    State('training-change-counter', 'data'),
    State('session-id', 'data')
)
def validate_training_text_input(value, previous_value, stored_train_file, counter, session_data):
    """
    This callback validates the input in the training data textbox.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    # Handle case where the text area is empty
    if not value:
        return {
            'width': '97.5%',
            'height': '100px'
        }, counter, previous_value

    # Check if input in the text area has changed
    if value != previous_value:
        counter += 1

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    proteins = []
    for row in rows:

        # Skip empty rows
        if row.strip() == "":
            continue
        row = row.strip()

        # Check different row formats
        if "|" in row:
            data = row.split("|")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                protein_value = float(protein_value)
            except ValueError:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        elif "," in row:
            data = row.split(",")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                float_protein_value = float(protein_value)
            except ValueError:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            protein_value = float_protein_value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        elif ";" in row:
            data = row.split(";")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                float_protein_value = float(protein_value)
            except ValueError:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }, counter, value

            protein_value = float_protein_value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        else:
            return {
                'width': '97.5%',
                'height': '100px',
                'border': '2px solid #dc3545'
            }, counter, value

    # Check if length of input is less than 5 (since we need at least 5 for 5-fold)
    if len(sequences) < 5 and len(proteins) < 5:
        return {
            'width': '97.5%',
            'height': '100px',
            'border': '2px solid #dc3545'
        }, counter, value

    # If the data was not set using the file upload, use the data in the textarea instead
    if not stored_train_file:
        user_data['TRAINING_FILE'] = f"training_text_input_{counter}"
        user_data['TRAINING_DATA'] = pd.DataFrame({
            'sequence': sequences,
            'protein': proteins
        }).to_json()
        globals.store_user_session_data(session_id, user_data)

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }, counter, value


@callback(
    [Output('upload-testing-data', 'children'),
     Output('store-uploaded-test-file', 'data')],
    Input('upload-testing-data', 'contents'),
    [State('upload-testing-data', 'filename'),
     State('store-uploaded-test-file', 'data'),
     State('session-id', 'data')]
)
def update_testing_output(content, name, stored_test_file_name, session_data):
    """
    This callback updates the contents of the upload box for the
    model test data.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    if stored_test_file_name and not content:
        file = stored_test_file_name['filename']

        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {file}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {file} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        final_display = html.Div([upload_children, success_message])
        return final_display, dash.no_update

    elif content:
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

        wrong_format_message = html.Div(
            [f"File {name} is not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        wrong_columns_message = html.Div(
            [f"File {name} use the wrong column names!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        invalid_data_message = html.Div(
            [f"File {name} uses data in the wrong format!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        limit_exceeded_message = html.Div(
            [f"File {name} is too large or sequences are too long!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Check if file is a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Check if file contains two columns labeled 'sequence' and 'protein'
            df.columns = df.columns.astype(str).str.lower()
            if 'sequence' in df.columns and 'protein' in df.columns:

                # Check data in the protein column is of the correct type
                parsed_protein = pd.to_numeric(df['protein'], errors='coerce')
                if df["protein"].dtype != 'float64' or not parsed_protein.notna().all():
                    final_display = html.Div([upload_children, invalid_data_message])
                    return final_display, None

                # Check data in the sequence column is of the correct type
                # we check that the data is a string, all sequences are the same length
                # and all sequences contain only a mix of the characters A, C, G, and T and nothing else
                invalid_sequences = df['sequence'].str.lower().str.contains('[^actg]')
                if df['sequence'].dtype != object or df['sequence'].str.len().nunique() != 1 or invalid_sequences.any():
                    final_display = html.Div([upload_children, invalid_data_message])
                    return final_display, None

                # For memory's sake, limit size file and the length of each sequence
                if len(df) > 20000 or df['sequence'].str.len()[0] > 250:
                    final_display = html.Div([upload_children, limit_exceeded_message])
                    return final_display, None

                df['sequence'] = df['sequence'].str.lower()
                final_display = html.Div([upload_children, success_message])

                # Set the training file name and data in REDIS for that user
                user_data['TESTING_FILE'] = name
                user_data['TESTING_DATA'] = df.to_json()
                globals.store_user_session_data(session_id, user_data)

                return final_display, {'filename': name}

            # If CSV but without right columns
            final_display = html.Div([upload_children, wrong_columns_message])
            return final_display, None

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, wrong_format_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-test-data', 'style'),
    Output('testing-change-counter', 'data'),
    Output('previous-test-value', 'value'),
    Input('text-test-data', 'value'),
    State('previous-test-value', 'value'),
    State('store-uploaded-test-file', 'data'),
    State('testing-change-counter', 'data'),
    State('session-id', 'data')
)
def validate_testing_text_input(value, previous_value, stored_test_file, counter, session_data):
    """
    This callback validates the input in the testing data textbox.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    # Handle case where the text area is empty
    if not value:
        return {
           'width': '97.5%',
           'height': '100px'
        }, counter, previous_value

    # Check if input in the text area has changed
    if value != previous_value:
        counter += 1

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    proteins = []
    for row in rows:

        # Skip empty rows
        if row.strip() == "":
            continue
        row = row.strip()

        # Check different row formats
        if "|" in row:
            data = row.split("|")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                protein_value = float(protein_value)
            except ValueError:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        elif "," in row:
            data = row.split(",")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                float_protein_value = float(protein_value)
            except ValueError:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            protein_value = float_protein_value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        elif ";" in row:
            data = row.split(";")

            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = data[0].strip()
            protein_value = data[1].strip()

            # check if protein value is valid
            try:
                float_protein_value = float(protein_value)
            except ValueError:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            protein_value = float_protein_value

            # Check if sequence value is valid

            # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
            allowed_chars = ['a', 'c', 't', 'g']
            if any(char not in allowed_chars for char in sequence_value.lower()):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check that all sequences are the same length
            if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            # check length of sequence does not exceed limit of 250 nt
            # and length of input does not exceed limit of 20,000
            if len(sequences[0]) > 250 and len(sequences) > 20000:
                return {
                   'width': '97.5%',
                   'height': '100px',
                   'border': '2px solid #dc3545'
                }, counter, value

            sequence_value = sequence_value.lower()
            # otherwise, data is valid
            sequences.append(sequence_value)
            proteins.append(protein_value)

        else:
            return {
               'width': '97.5%',
               'height': '100px',
               'border': '2px solid #dc3545'
            }, counter, value

    # If the data was not set using the file upload, use the data in the textarea instead
    if not stored_test_file:
        user_data['TESTING_FILE'] = f"testing_text_input_{counter}"
        user_data['TESTING_DATA'] = pd.DataFrame({
            'sequence': sequences,
            'protein': proteins
        }).to_json()
        globals.store_user_session_data(session_id, user_data)

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }, counter, value


@callback(
    [Output('upload-querying-data', 'children'),
     Output('store-uploaded-query-file', 'data')],
    Input('upload-querying-data', 'contents'),
    [State('upload-querying-data', 'filename'),
     State('store-uploaded-query-file', 'data'),
     State('session-id', 'data')]
)
def update_querying_output(content, name, stored_query_file_name, session_data):
    """
    This callback updates the contents of the upload box for the
    model querying data.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    if stored_query_file_name and not content:
        file = stored_query_file_name['filename']

        # Update the dcc.Upload children to show the uploaded file's name
        upload_children = html.Div(
            [f"Uploaded: {file}"],
            style={
                "margin-top": "30px",
                "text-align": "center",
                "font-weight": "bold",
                "font-size": "12pt"
            }
        )

        success_message = html.Div(
            [f"File {file} uploaded successfully!"],
            style={
                "font-weight": "bold",
                "color": "green",
                "font-size": "12pt"
            }
        )

        final_display = html.Div([upload_children, success_message])
        return final_display, dash.no_update

    elif content:
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

        wrong_format_message = html.Div(
            [f"File {name} is not compatible!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        wrong_columns_message = html.Div(
            [f"File {name} use the wrong column names!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        invalid_data_message = html.Div(
            [f"File {name} uses data in the wrong format!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        limit_exceeded_message = html.Div(
            [f"File {name} is too large or sequences are too long!"],
            style={
                "font-weight": "bold",
                "color": "red",
                "font-size": "12pt"
            }
        )

        # Check if file is a CSV file
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        if '.csv' in name:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Check if file contains a column labeled 'sequence'
            df.columns = df.columns.astype(str).str.lower()
            if 'sequence' in df.columns:

                # Check data in the sequence column is of the correct type
                # we check that the data is a string, all sequences are the same length
                # and all sequences contain only a mix of the characters A, C, G, and T and nothing else
                invalid_sequences = df['sequence'].str.lower().str.contains('[^actg]')
                if df['sequence'].dtype != object or df['sequence'].str.len().nunique() != 1 or invalid_sequences.any():
                    final_display = html.Div([upload_children, invalid_data_message])
                    return final_display, None

                # For memory's sake, limit size file and the length of each sequence
                if len(df) > 20000 or df['sequence'].str.len()[0] > 250:
                    final_display = html.Div([upload_children, limit_exceeded_message])
                    return final_display, None

                df['sequence'] = df['sequence'].str.lower()
                final_display = html.Div([upload_children, success_message])

                # Set the training file name and data in REDIS for that user
                user_data['QUERYING_FILE'] = name
                user_data['QUERYING_DATA'] = df.to_json()
                globals.store_user_session_data(session_id, user_data)

                return final_display, {'filename': name}

            # If CSV but without right columns
            final_display = html.Div([upload_children, wrong_columns_message])
            return final_display, None

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, wrong_format_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-query-data', 'style'),
    Output('querying-change-counter', 'data'),
    Output('previous-query-value', 'value'),
    Input('text-query-data', 'value'),
    State('previous-query-value', 'value'),
    State('store-uploaded-query-file', 'data'),
    State('querying-change-counter', 'data'),
    State('session-id', 'data')
)
def validate_querying_text_input(value, previous_value, stored_query_file, counter, session_data):
    """
    This callback validates the input in the querying data textbox.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)

    # Handle case where the text area is empty
    if not value:
        return {
           'width': '97.5%',
           'height': '100px'
        }, counter, previous_value

    # Check if input in the text area has changed
    if value != previous_value:
        counter += 1

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    for row in rows:

        # Skip empty rows
        if row.strip() == "":
            continue
        sequence_value = row.strip()

        # Check if sequence value is valid

        # check all sequences contain only a mix of the characters A, C, G, and T and nothing else
        allowed_chars = ['a', 'c', 't', 'g']
        if any(char not in allowed_chars for char in sequence_value.lower()):
            return {
               'width': '97.5%',
               'height': '100px',
               'border': '2px solid #dc3545'
            }, counter, value

        # check that all sequences are the same length
        if len(sequences) >= 1 and len(sequences[0]) != len(sequence_value):
            return {
               'width': '97.5%',
               'height': '100px',
               'border': '2px solid #dc3545'
            }, counter, value

        # check length of sequence does not exceed limit of 250 nt
        # and length of input does not exceed limit of 20,000
        if len(sequences[0]) > 250 and len(sequences) > 20000:
            return {
               'width': '97.5%',
               'height': '100px',
               'border': '2px solid #dc3545'
            }, counter, value

        sequence_value = sequence_value.lower()
        # Otherwise, data is valid
        sequences.append(sequence_value)

    # If the data was not set using the file upload, use the data in the textarea instead
    if not stored_query_file:
        user_data['QUERYING_FILE'] = f"querying_text_input_{counter}"
        user_data['QUERYING_DATA'] = pd.DataFrame({
            'sequence': sequences,
        }).to_json()
        globals.store_user_session_data(session_id, user_data)

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }, counter, value


@callback(
    Output('content', 'children'),
    Input('container', 'value'),
    State('store-model-count', 'data'),
    State('session-id', 'data')
)
def render_tabs_content(selected_tab, stored_count, session_data):
    """
    This callback function keeps track of the user changes to the
    tabs container (and displays the correct information for each tab)
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)
    models_list = user_data['MODELS_LIST']

    # File upload tab
    if selected_tab == 'upload datasets':

        return dbc.Container(
            id='tabs-content-upload',
            children=[
                html.Div(id='uploaded-files'),
                dbc.Row(
                    id='card-row',
                    children=[
                        dbc.Col(
                            children=[training_data_upload_card()],
                            md=3,
                            style={
                                'margin-left': '50px',
                                'margin-right': '50px'
                            }
                        ),
                        dbc.Col(
                            children=[testing_data_upload_card()],
                            md=5,
                            style={
                                'margin-left': '50px',
                                'margin-right': '50px'
                            }
                        ),
                        dbc.Col(
                            children=[query_data_upload_card()],
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
        if stored_count:
            return dbc.Row(
                id='tabs-content-input',
                children=[model_input_ref(model_key, session_id) for model_key in models_list.keys()] +
                         [
                             html.Button(
                                 'Add a new model',
                                 id='button',
                                 n_clicks=stored_count['n_clicks']
                             )
                         ]
            )
        # Initial state
        else:
            return dbc.Row(
                id='tabs-content-input',
                children=[
                    # This creates the initial layout with one model
                    model_input_ref("Model 1", session_id),
                    html.Button(
                        'Add a new model',
                        id='button',
                        n_clicks=1
                    )
                ]
            )

    # Model outputs
    elif selected_tab == "model outputs":
        return dbc.Row(
            id='tabs-content-output',
            children=[output_statistics_card(), html.Div(id='table-container')] +
                     [model_output_ref(model_key) for model_key in models_list.keys()]
        )

    # Validation check
    else:
        return 'No content available.'


@callback(
    [Output('tabs-content-input', 'children'),
     Output('store-model-count', 'data')
     ],
    Input('button', 'n_clicks'),
    [State('tabs-content-input', 'children'),
     State('store-model-count', 'data'),
     State('session-id', 'data')
     ]
)
def add_new_model_tab(n_clicks, current_children, stored_count, session_data):
    """
    This callback function keeps track of the user changes to the
    model inputs tab (when adding new models).
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']

    model_key = f'Model {n_clicks}'

    # Check if a new model has been added
    if n_clicks > stored_count['n_clicks']:
        stored_count['n_clicks'] = n_clicks
        children = current_children + [model_input_ref(model_key, session_id)]
        return children, stored_count

    # If there has been no new model added
    return dash.no_update, dash.no_update


@callback(
    Output('page-content', 'children'),
    Input('url', 'href'),
    Input('session-id', 'data')
)
def display_page(href, session_data):
    """
    This callback allows for switching between tabs when choosing to view
    individual model inputs/ outputs.
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data['session_id']
    user_data = globals.get_user_session_data(session_id)
    models_list = user_data['MODELS_LIST']

    # Extract pathname from the full URL (href)
    parsed_url = urlparse(href)
    pathname = parsed_url.path

    if pathname.startswith('/model-input/'):
        # If a model inputs tab is selected, return the card for that input
        try:
            model_num = int(pathname.split('/')[-1][-1])
            model_key = f"Model {model_num}"
            if model_key in models_list.keys():
                return model_inputs_page.create_layout(model_num)
            else:
                return html.Div('Invalid model number.')
        except ValueError:
            return html.Div('Invalid URL.')

    elif pathname.startswith('/model-output/'):
        # If a model output tab is selected, return the card for that output
        try:
            model_num = int(pathname.split('/')[-1][-1])
            model_key = f"Model {model_num}"
            if model_key in models_list.keys():
                return model_outputs_page.create_layout(model_num, session_data)
            else:
                return html.Div('Invalid model number.')
        except ValueError:
            return html.Div('Invalid URL.')

    elif pathname.startswith('/output-statistics/'):
        # If the output statistics page is created
        return output_statistics_page.create_layout()

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
                        app_footer()
                    ]
                )
            ]
        )
    ]


# Update individual tab titles depending on the page
clientside_callback(
    "dash_clientside.clientside.updateTitle",
    Output('page-title', 'children'),
    Input('url', 'href')
)


@server.route('/downloadable_data/<path:filename>')
def download_training_file(filename):
    """
    This callback correctly downloads all the example CSV files for the server.
    """

    directory = os.path.join(os.getcwd(), 'downloadable_data')
    return send_from_directory(directory, filename, as_attachment=True)


@callback(
    Output('session-id', 'data'),
    [Input('url', 'pathname')]
)
def create_or_fetch_session_id(_pathname):
    """
    This callback sets the session ID for each user
    """

    # Check if the session ID already exists
    if 'user_session_id' not in session:
        # Generate a new session ID if it does not exist
        session['user_session_id'] = str(uuid.uuid4())
        # Initialize empty values in Redis for this session ID
        data = {
            'MODELS_LIST': {'Model 1': None},
            'TRAINING_DATA': None,
            'TRAINING_FILE': None,
            'TESTING_DATA': None,
            'TESTING_FILE': None,
            'QUERYING_DATA': None,
            'QUERYING_FILE': None
        }
        globals.store_user_session_data(session['user_session_id'], data)

    try:
        globals.get_user_session_data(session['user_session_id'])
    except Exception:
        # Initialize empty values in Redis for this session ID
        data = {
            'MODELS_LIST': {'Model 1': None},
            'TRAINING_DATA': None,
            'TRAINING_FILE': None,
            'TESTING_DATA': None,
            'TESTING_FILE': None,
            'QUERYING_DATA': None,
            'QUERYING_FILE': None
        }
        globals.store_user_session_data(session['user_session_id'], data)

    # Return the session ID
    return {'session_id': session['user_session_id']}


def cleanup_old_session_data():
    """
    This function removes any entries in the REDIS database that have
    been inactive for a significant period of time.
    """

    current_time = int(time.time())
    time_limit = 1800

    # Iterate over timestamp keys to find old data
    for key in globals.redis_client.scan_iter("session:timestamp:*"):
        # Check if key hasn't been accessed within the time limit
        last_access_time = int(globals.redis_client.get(key))
        if current_time - last_access_time > time_limit:
            # Extract session_id from the key
            session_id = key.split(":")[-1]
            # Delete both the data and timestamp
            globals.redis_client.delete(f"session:data:{session_id}")
            globals.redis_client.delete(f"session:timestamp:{session_id}")


# Schedule the cleanup task
scheduler.add_job(id='Cleanup Old Session Data', func=cleanup_old_session_data, trigger='interval', minutes=30)

# Main layout of the home page
my_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-title', style={'display': 'none'}),
    html.Div(id='page-content'),
    dcc.Store(id='store-model-count', data={'n_clicks': 1}, storage_type='session'),
    dcc.Store(id='store-uploaded-train-file', storage_type='session'),
    dcc.Store(id='store-uploaded-test-file', storage_type='session'),
    dcc.Store(id='store-uploaded-query-file', storage_type='session'),
    dcc.Store(id='training-change-counter', data=0, storage_type='session'),
    dcc.Store(id='testing-change-counter', data=0, storage_type='session'),
    dcc.Store(id='querying-change-counter', data=0, storage_type='session'),
    dcc.Store(id='session-id', storage_type='session')
])

if __name__ == '__main__':
    my_app.run_server(port=int(os.environ.get("PORT", 8050)))
