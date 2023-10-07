import base64
from dash import Dash, html

app = Dash(__name__)

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
            id='app-page-content'
        )
    ],
)

if __name__ == '__main__':
    app.run_server()
