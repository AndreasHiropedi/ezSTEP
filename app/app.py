import base64
from dash import Dash, html, dcc, Input, Output, callback

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    children=[
        # Web app header
        html.Header(
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
        ),

        # Main page content
        html.Div(
            id='app-contents',
            children=[
                # Text part explaining how the app works
                html.Div(
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
                                                "   1. Model input parameters",
                                                html.Br(),
                                                html.Br(),
                                                "   2. File upload",
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
                                        'border-color': '#f5c6cb'
                                    }
                                ),

                                # Model input parameters
                                html.Div(
                                    id='model-input',
                                    children=[
                                        html.H4("2. Model input parameters"),
                                        html.Hr(),
                                        html.P(
                                            "In this section, the user gets to select a model and input all the "
                                            "necessary information in order to train and test that model. This "
                                            "information will be used together with the datasets uploaded (see next "
                                            "section for more details) in order to train the models and visualise the "
                                            "output (see the 'Model Output Visualisations' section for more). The user "
                                            "will also be able to add more than one model, but will need to input all "
                                            "the necessary for each model added. There will also be an option that "
                                            "will allow the user to choose whether or not they wish to optimise the "
                                            "model's hyperparameters or not."
                                        )
                                    ],
                                    style={
                                        'background': '#cce5ff',
                                        'color': '#004085',
                                        'border-color': '#b8daff'
                                    }
                                ),

                                # File upload
                                html.Div(
                                    id='file-upload',
                                    children=[
                                        html.H4("3. File upload"),
                                        html.Hr(),
                                        html.P(
                                            "This section contains three upload boxes, two of which are required "
                                            "fields, and one optional field. The two required fields are for uploading "
                                            "the training data (in order to train the selected model) and the testing "
                                            "data (in order to test the trained model). The optional field is for "
                                            "uploading a dataset for querying the model on new unseen data that was "
                                            "not part of the training or testing dataset. If the user has added more "
                                            "than one model, then for each model they added, they can select if they "
                                            "wish to use the same datasets, or upload new ones instead."
                                        )
                                    ],
                                    style={
                                        'background': '#e2e3e5',
                                        'color': '#383d41',
                                        'border-color': '#d6d8db'
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
                                            "all models and their RMSEs will be generated."
                                        )
                                    ],
                                    style={
                                        'background': '#fff3cd',
                                        'color': '#856404',
                                        'border-color': '#ffeeba'
                                    }
                                ),
                            ]
                        )
                    ]
                ),

                # Input tabs
                html.Div(
                    id='tabs-container',
                    children=[
                        dcc.Tabs(
                            id='container',
                            value="model input parameters",
                            children=[
                                dcc.Tab(
                                    id='tab-input',
                                    label="Model input parameters",
                                    value="model input parameters"
                                ),
                                dcc.Tab(
                                    id='tab-upload',
                                    label="Upload datasets",
                                    value="upload datasets"
                                ),
                                dcc.Tab(
                                    id='tab-visualise',
                                    label="Visualise model outputs",
                                    value="visualise model outputs"
                                )
                            ]
                        ),

                        # Output of the tabs
                        html.Div(
                            id='tabs-content',
                            children=[
                                html.Div(
                                    id='content'
                                )
                            ]
                        )
                    ]
                )
            ],
            style={
                'background': '#2E2B2A',
                'color': 'white'
            }
        ),

        # Web app footer
        html.Footer(

        )
    ],
)


@callback(Output('content', 'children'),
          Input('container', 'value'))
def render_content(tab):
    if tab == 'model input parameters':
        return html.H3('User input tabs')
    elif tab == 'upload datasets':
        return html.H3('Upload boxes for datasets')
    elif tab == "visualise model outputs":
        return html.H3('Model outputs')
    else:
        return 'No content available.'


if __name__ == '__main__':
    app.run_server()
