import dash_bootstrap_components as dbc
import plotly.express as px

from dash import html, dcc, callback, Input, Output, MATCH


def create_layout(model_count):
    """
    This function creates all pages for the model outputs.
    """

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

                ]
            )
        ]
    )


