import base64
import pickle

import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from database import db


def create_layout():
    """
    This function creates the main graph for the testing output statistics.
    """

    return html.Div(
        id="output-statistics-page",
        children=[
            html.Div(
                id="statistics-page-header",
                children=[html.H1(children=["Testing output statistics graph"])],
            ),
            html.Div(
                id="statistics-page-contents",
                children=[dcc.Graph(id="statistics-graph", style={"height": "600px"})],
            ),
        ],
    )


@callback(
    Output("statistics-graph", "figure"),
    Input("statistics-graph", "id"),
    State("session-id", "data"),
)
def update_statistics_graph(_id, session_data):
    """
    This callback generates the output statistics graph based on the existing models
    """

    # Get the session ID for that user, and the data in REDIS
    session_id = session_data["session_id"]
    user_data = db.get_user_session_data(session_id)
    models_list = user_data["MODELS_LIST"]

    # only stores the created models (in case the user adds more models but doesn't initialise all of them)
    # to avoid errors when plotting the graph
    existing_models = {}
    for key, value in models_list.items():
        if value is not None:
            model_bytes = base64.b64decode(value)
            existing_models[key] = pickle.loads(model_bytes)

    figure = go.Figure()

    categories = ["RMSE", "R-squared", "MAE", "Two-fold error", "RMSE"]

    # Add all data for each model to the figure
    for model_name, model in existing_models.items():
        data = [
            model.testing_RMSE,
            model.testing_R_squared,
            model.testing_MAE,
            model.testing_2fold_error,
            model.testing_RMSE,
        ]

        model_type = model.model_name
        model_label = model_name + " - " + model_type

        figure.add_trace(go.Scatterpolar(r=data, theta=categories, name=model_label))

    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])), showlegend=True
    )

    return figure
