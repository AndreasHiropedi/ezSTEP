import base64
from dash import Dash, html, dcc, Input, Output, callback

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    children=[
        # Web app header
        html.Div(
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
            children=[
                # Text part explaining how the app works
                html.Div(
                    id='user-guide',
                    children=[
                        html.H3("Here is how to use the app")
                    ]
                ),

                # Input tabs
                dcc.Tabs(
                    id='tabs-container',
                    className='tabs-container',
                    value="model input parameters",
                    children=[
                        dcc.Tab(
                            id='tab',
                            label="Model input parameters",
                            value="model input parameters"
                        ),
                        dcc.Tab(
                            id='tab',
                            label="Upload datasets",
                            value="upload datasets"
                        ),
                        dcc.Tab(
                            id='tab',
                            label="Visualize model outputs",
                            value="visualize model outputs"
                        )
                    ]
                ),

                # Output of the tabs
                html.Div(id='tabs-content')
            ],
        )
    ],
)


@callback(Output('tabs-content', 'children'),
          Input('tabs-container', 'value'))
def render_content(tab):
    if tab == 'model input parameters':
        return html.Div([
            html.H3('User input tabs')
        ])
    elif tab == 'upload datasets':
        return html.Div([
            html.H3('Upload boxes for datasets')
        ])
    elif tab == "visualize model outputs":
        return html.Div([
            html.H3('Model outputs')
        ])
    return 'No content available.'


if __name__ == '__main__':
    app.run_server()
