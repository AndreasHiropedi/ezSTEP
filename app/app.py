import base64
import dash_bootstrap_components as dbc
import dash
from dash import Dash, html, dcc, callback, Input, Output, State

app = Dash(__name__, suppress_callback_exceptions=True)


def app_header():
    # Web app header
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
                "DNA Sequence Analyser",
            ),

            # GitHub repo link
            html.A(
                id='github-link',
                children=[
                    "View on GitHub"
                ],
                href='https://github.com/AndreasHiropedi/Dissertation',
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
    # Web app footer
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
    # User guide explaining how the app works
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
                                "This section contains three upload boxes, one of which is a required "
                                "field, and two optional fields. The required field is for uploading "
                                "the training data (in order to train the selected model). The optional "
                                "fields are for uploading the testing data (in case the user doesn't wish "
                                "to split the training dataset) and for uploading a dataset for querying "
                                "the model on new unseen data that was not part of the training or testing "
                                "dataset. If the user has added more than one model, then for each model "
                                "they added, they can select if they wish to use the same datasets, "
                                "or upload new ones instead."
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
                                "information will be used together with the datasets uploaded (see next "
                                "section for more details) in order to train the models and visualise the "
                                "output (see the 'Model Output Visualisations' section for more). The user "
                                "will also be able to add more than one model, but will need to input all "
                                "the necessary information for each model added. There will also be an  "
                                "option that will allow the user to choose whether or not they wish to  "
                                "optimise the model's hyperparameters or not."
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


def training_data_upload_card():
    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Training data"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your training data",
                        style={
                            'text-align': 'center',
                            'font-size': '16pt'
                        }
                    ),
                    dcc.Upload(
                        children=[
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
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
                            'font-size': '16pt'
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
        style={
            'background': 'white',
            'color': 'black',
            'width': '65%',
            'margin-top': '30px',
            'border': '2px solid black',
            'margin-left': '250px',
        }
    )


def testing_data_upload_card():
    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Testing data (optional)"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your testing data",
                        style={
                            'text-align': 'center',
                            'font-size': '16pt'
                        }
                    ),
                    dcc.Upload(
                        children=[
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
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
                            'font-size': '16pt'
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
        style={
            'background': 'white',
            'color': 'black',
            'width': '65%',
            'margin-top': '30px',
            'border': '2px solid black',
            'margin-left': '250px',
        }
    )


def query_data_upload_card():
    return dbc.Card(
        id='card',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Model querying data (optional)"],
            ),
            dbc.CardBody(
                id='card-body',
                children=[
                    html.P(
                        "Upload your model querying data",
                        style={
                            'text-align': 'center',
                            'font-size': '16pt'
                        }
                    ),
                    dcc.Upload(
                        children=[
                            html.Div(
                                id='box-text',
                                children=[
                                    'Drag and Drop or ',
                                    html.A('Select Files', style={'font-weight': 'bold'})
                                ],
                            )
                        ],
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
                            'font-size': '16pt'
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
        style={
            'background': 'white',
            'color': 'black',
            'width': '65%',
            'margin-top': '30px',
            'border': '2px solid black',
            'margin-left': '250px',
        }
    )


def initial_model_input_parameters_card():
    return dbc.Card(
        id='card-input',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=["Model 1 parameters"]
            ),
            dbc.CardBody(
                id='card-body',
                children=["To fill in with input parameters"]
            )
        ]
    )


def extra_model_input_parameters_card(model_count):
    return dbc.Card(
        id='card-input',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=[f"Model {model_count} parameters"]
            ),
            dbc.CardBody(
                id='card-body',
                children=["To fill in with input parameters"]
            )
        ]
    )


def model_output_card(model_count):
    return dbc.Card(
        id='card-output',
        children=[
            dbc.CardHeader(
                id='card-header',
                children=[f"Model {model_count} output"]
            ),
            dbc.CardBody(
                id='card-body',
                children=["To fill in with output visualisations"]
            )
        ]
    )


@callback(
    Output('content', 'children'),
    Input('container', 'value'),
    State('store-model-content', 'data')
)
def render_tabs_content(selected_tab, stored_content):
    # File upload tab
    if selected_tab == 'upload datasets':
        return html.Div(
            id='tabs-content-upload',
            children=[
                training_data_upload_card(),
                testing_data_upload_card(),
                query_data_upload_card()
            ],
            style={
                'background': 'white',
                'color': 'black'
            }
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
                    html.Button(
                        'Add a new model',
                        id='button',
                        n_clicks=1
                    ),
                    # This allows more models to be added and stored once added
                    html.Div(
                        id='button-output',
                        children=[initial_model_input_parameters_card()]
                    )
                ]
            )

    # Model outputs
    elif selected_tab == "visualise model outputs":
        return dbc.Row(
            id='tabs-content-output',
            children=[model_output_card(1)]
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
    # Check if a new model has been added
    if n_clicks > stored_count['n_clicks']:
        stored_count['n_clicks'] = n_clicks
        children = current_children + [extra_model_input_parameters_card(n_clicks)]
        new_content = {
            'children': children
        }
        return children, stored_count, new_content

    # If there has been no new model added
    return dash.no_update, dash.no_update, dash.no_update


app.layout = html.Div(
    children=[
        app_header(),
        # Body of the page
        html.Div(
            id='app-contents',
            children=[
                user_guide(),
                # Tabs
                html.Div(
                    id='tabs-container',
                    children=[
                        tabs_container(),
                        # Output of the tabs
                        html.Div(
                            id='content'
                        ),
                        dcc.Store(
                            id='store-model-count',
                            data={'n_clicks': 1}
                        ),
                        dcc.Store(
                            id='store-model-content',
                            data={}
                        ),
                        app_footer()
                    ],
                )
            ],
            style={
                'background': 'white',
                'color': 'black'
            }
        ),
    ],
)

if __name__ == '__main__':
    app.run_server()
