import base64
import dash
import dash_bootstrap_components as dbc
import io
import pandas as pd
import os

from app.app_init import my_app
from dash import html, dcc, callback, Input, Output, State, clientside_callback
from pages import model_inputs_page, model_outputs_page, output_statistics_page
from urllib.parse import urlparse


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
                            ),
                            html.P(
                                "NOTE: for the training process, we always perform 5-fold cross validation on the "
                                "uploaded training dataset."
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
                                "output (see the 'Model Outputs' section for more). The user "
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
                            html.H4("4. Model outputs"),
                            html.Hr(),
                            html.P(
                                "Once the data has been uploaded and the user has set all the input "
                                "parameters, the visualisations for the specific model, along with some "
                                "statistics (such as the root mean squared error (RMSE)) are generated. "
                                "If the user has added several models, then the visualisations for each "
                                "model will be displayed on the model's own page, and a table showing "
                                "all models and their performance statistics will be displayed on the main page. "
                            ),
                            html.P(
                                "Similar to the model inputs, in order to see the visualisations for each "
                                "individual model created, the user will need to click on the appropriate "
                                "hyperlink, which will prompt them to the correct page. However, it is worth "
                                "noting that the table containing all the models and the selected performance "
                                "statistics will be displayed on the main page."
                            ),
                            html.P(
                                "If the user has uploaded a querying dataset, the output of the model on that "
                                "dataset will also be made available on the individual hyperlink pages. This will be "
                                "in the form of a hyperlink that will allow users to download a CSV file containing "
                                "the model's output on the querying dataset."
                            ),
                            html.P(
                                children=[
                                    "The user will be able to select the summary statistics; these statistics will "
                                    "be displayed individually using bar charts (to access these charts, the user "
                                    "will need to click on the respective hyperlink which is automatically generated"
                                    "when a statistic is selected from the dropdown. The four main summary statistics "
                                    "available are: ",
                                    html.Br(),
                                    html.Br(),
                                    "1. Root Mean Squared Error (RMSE)",
                                    html.Br(),
                                    html.Br(),
                                    "2. R-squared ",
                                    html.Br(),
                                    html.Br(),
                                    "3. Mean Absolute Error (MAE)",
                                    html.Br(),
                                    html.Br(),
                                    "4. Percentage (%) within 2-fold error (this essentially measures the proportion of"
                                    " data that are within the 2-fold error interval: 1/2 × correct value ≤ "
                                    "predicted value ≤ 2 × correct value)"
                                ]
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
                    )
                ]
            )
        ],
        className="mx-auto"
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
                "Select the evaluation statistics (for testing) about the model that you wish to see in the output  "
                "(select all that apply):",
                id='select-statistics'
            ),
            dcc.Dropdown(
                id='output-statistics-dropdown',
                options=[
                    {'label': 'RMSE', 'value': 'RMSE'},
                    {'label': 'R-squared', 'value': 'R-squared'},
                    {'label': 'MAE', 'value': 'MAE'},
                    {'label': 'Percentage within 2-fold error', 'value': 'Percentage within 2-fold error'},
                ],
                value=[],
                multi=True,
                searchable=False,
                persistence=True,
                persistence_type='session'
            )
        ]
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
                children=[output_statistics_dropdown()]
            ),
            html.Div(id='hyperlinks-container')
        ]
    )


def model_input_ref(model_count):
    """
    This function creates hyperlinks to a separate page for each
    model input (for each model created).
    """

    return html.A(
        children=[
            html.H4(
                f"Model {model_count} input parameters",
                id='model-inputs-ref'
            )
        ],
        href=f'/model-input/model-{model_count}',
        target='_blank'
    )


def output_metric_ref(metric_name):
    """
    This function creates hyperlinks to a separate page for each
    output statistic selected by the user (using the dropdown in the output tab).
    """

    return html.A(
        children=[
            html.H4(
                f"View {metric_name} plot",
                id='metric-plot-ref'
            )
        ],
        href=f'/output-statistics/{metric_name}',
        target='_blank'
    )


def model_output_ref(model_count):
    """
    This function creates hyperlinks to a separate page for each
    model output (for each model created).
    """

    return html.A(
        children=[
            html.H4(
                f"Model {model_count} output",
                id='model-outputs-ref'
            )
        ],
        href=f'/model-output/model-{model_count}',
        target='_blank'
    )


@callback(
    [Output('upload-training-data', 'children'),
     Output('store-uploaded-train-file', 'data')],
    Input('upload-training-data', 'contents'),
    [State('upload-training-data', 'filename'),
     State('store-uploaded-train-file', 'data')]
)
def update_training_output(content, name, stored_train_file_name):
    """
    This callback updates the contents of the upload box for the
    model train data.
    """

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
            _df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-train-data', 'style'),
    Input('text-train-data', 'value')
)
def validate_training_text_input(value):
    """

    """

    # Handle case where the text area is empty
    if not value:
        return {
            'width': '97.5%',
            'height': '100px'
        }

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    labels = []
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
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif "," in row:
            data = row.split(",")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif ";" in row:
            data = row.split(";")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        else:
            return {
                'width': '97.5%',
                'height': '100px',
                'border': '2px solid #dc3545'
            }

    # TODO: ADD CHECK IF FILE WAS UPLOADED (AND KEEP THAT DATA), OTHERWISE USE THIS VALID DATA

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }


@callback(
    [Output('upload-testing-data', 'children'),
     Output('store-uploaded-test-file', 'data')],
    Input('upload-testing-data', 'contents'),
    [State('upload-testing-data', 'filename'),
     State('store-uploaded-test-file', 'data')]
)
def update_testing_output(content, name, stored_test_file_name):
    """
    This callback updates the contents of the upload box for the
    model test data.
    """

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
            _df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-test-data', 'style'),
    Input('text-test-data', 'value')
)
def validate_testing_text_input(value):
    """

    """

    # Handle case where the text area is empty
    if not value:
        return {
            'width': '97.5%',
            'height': '100px'
        }

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    labels = []
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
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif "," in row:
            data = row.split(",")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif ";" in row:
            data = row.split(";")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        else:
            return {
                'width': '97.5%',
                'height': '100px',
                'border': '2px solid #dc3545'
            }

    # TODO: ADD CHECK IF FILE WAS UPLOADED (AND KEEP THAT DATA), OTHERWISE USE THIS VALID DATA

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }


@callback(
    [Output('upload-querying-data', 'children'),
     Output('store-uploaded-query-file', 'data')],
    Input('upload-querying-data', 'contents'),
    [State('upload-querying-data', 'filename'),
     State('store-uploaded-query-file', 'data')]
)
def update_querying_output(content, name, stored_query_file_name):
    """
    This callback updates the contents of the upload box for the
    model querying data.
    """

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
            _df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            final_display = html.Div([upload_children, success_message])
            return final_display, {'filename': name}

        # If non-CSV, you can just return the name and a message or other placeholder
        final_display = html.Div([upload_children, failed_message])
        return final_display, None

    # If no content, revert to original children for dcc.Upload
    return html.Div(
        id='box-text',
        children=['Drag and Drop or ', html.A('Select Files', style={'font-weight': 'bold'})]), None


@callback(
    Output('text-query-data', 'style'),
    Input('text-query-data', 'value')
)
def validate_querying_text_input(value):
    """

    """

    # Handle case where the text area is empty
    if not value:
        return {
            'width': '97.5%',
            'height': '100px'
        }

    # Split the text into rows
    rows = value.split('\n')

    sequences = []
    labels = []
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
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif "," in row:
            data = row.split(",")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        elif ";" in row:
            data = row.split(";")
            # if there aren't exactly 2 elements, the data is invalid
            if len(data) != 2:
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # if one of them is empty, the data is invalid
            if data[0].strip() == '' or data[1].strip() == '':
                return {
                    'width': '97.5%',
                    'height': '100px',
                    'border': '2px solid #dc3545'
                }
            # otherwise, data is valid
            sequences.append(data[0])
            labels.append(data[1])

        else:
            return {
                'width': '97.5%',
                'height': '100px',
                'border': '2px solid #dc3545'
            }

    # TODO: ADD CHECK IF FILE WAS UPLOADED (AND KEEP THAT DATA), OTHERWISE USE THIS VALID DATA

    return {
        'width': '97.5%',
        'height': '100px',
        'border': '2px solid #28a745'
    }


@callback(
    Output('hyperlinks-container', 'children'),
    Input('output-statistics-dropdown', 'value')
)
def generate_hyperlinks(values_list):
    """
    This callback function generates the hyperlinks for testing statistics
    visualisations based on the user inputs in the dropdown.
    """

    visualizations_links = [output_metric_ref(value) for value in values_list]

    return visualizations_links


@callback(
    Output('content', 'children'),
    Input('container', 'value'),
    [State('store-model-count', 'data')]
)
def render_tabs_content(
        selected_tab,
        stored_count
):
    """
    This callback function keeps track of the user changes to the
    tabs container (and displays the correct information for each tab)
    """

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
                children=[model_input_ref(i) for i in range(1, stored_count['n_clicks'] + 1)] +
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
                    model_input_ref("1"),
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
                     [model_output_ref(i) for i in range(1, stored_count['n_clicks'] + 1)]
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
     State('store-model-count', 'data')
     ]
)
def add_new_model(n_clicks, current_children, stored_count):
    """
    This callback function keeps track of the user changes to the
    model inputs tab (when adding new models).
    """

    # Check if a new model has been added
    if n_clicks > stored_count['n_clicks']:
        stored_count['n_clicks'] = n_clicks
        children = current_children + [model_input_ref(n_clicks)]
        return children, stored_count

    # If there has been no new model added
    return dash.no_update, dash.no_update


@callback(
    Output('page-content', 'children'),
    Input('url', 'href')
)
def display_page(href):
    """
    This callback allows for switching between tabs when choosing to view
    individual model inputs/ outputs.
    """

    # Extract pathname from the full URL (href)
    parsed_url = urlparse(href)
    pathname = parsed_url.path

    if pathname.startswith('/model-input/'):
        # If a model inputs tab is selected, return the card for that input
        try:
            model_num = int(pathname.split('/')[-1][-1])
            return model_inputs_page.create_layout(model_num)
        except ValueError:
            return html.Div('Invalid model number.')

    elif pathname.startswith('/model-output/'):
        # If a model output tab is selected, return the card for that output
        try:
            model_num = int(pathname.split('/')[-1][-1])
            return model_outputs_page.create_layout(model_num)
        except ValueError:
            return html.Div('Invalid model number.')

    elif pathname.startswith('/output-statistics/'):
        # If one of the output statistics tabs is selected, check which one and display the appropriate graph
        if 'RMSE' in pathname:
            return output_statistics_page.create_layout('RMSE')

        elif 'R-squared' in pathname:
            return output_statistics_page.create_layout('R-squared')

        elif 'MAE' in pathname:
            return output_statistics_page.create_layout('MAE')

        # needed to format this way due to how browsers render this path
        elif 'Percentage%20within%202-fold%20error' in pathname:
            return output_statistics_page.create_layout('Percentage within 2-fold error')

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


my_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-title'),
    html.Div(id='page-content'),
    dcc.Store(id='store-model-count', data={'n_clicks': 1}, storage_type='session'),
    dcc.Store(id='store-uploaded-train-file', storage_type='session'),
    dcc.Store(id='store-uploaded-test-file', storage_type='session'),
    dcc.Store(id='store-uploaded-query-file', storage_type='session')
])

if __name__ == '__main__':
    my_app.run_server(port=int(os.environ.get("PORT", 8050)))
