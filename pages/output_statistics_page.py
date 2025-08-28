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
            # Header section with responsive styling
            html.Div(
                id="statistics-page-header",
                className="responsive-user-info",
                children=[
                    html.H1(
                        children=["Testing Output Statistics"],
                        className="responsive-user-info-title",
                        style={
                            "textAlign": "center",
                            "marginBottom": "2rem",
                            "color": "#333",
                        },
                    )
                ],
            ),
            # Graph container with responsive card styling
            html.Div(
                id="statistics-page-contents",
                className="responsive-output-statistics-card",
                children=[
                    html.Div(
                        className="centered-card-body",
                        children=[
                            html.P(
                                "Compare the performance metrics of your trained models using this radar chart visualization.",
                                className="centered-text",
                                style={
                                    "fontSize": "clamp(14px, 1.4vw, 18px)",
                                    "lineHeight": "1.6",
                                    "color": "#666",
                                    "marginBottom": "2rem",
                                },
                            ),
                            dcc.Graph(
                                id="statistics-graph",
                                style={
                                    "width": "100%",
                                    "height": "clamp(700px, 80vh, 1000px)",
                                    "minHeight": "600px",
                                },
                                config={
                                    "displayModeBar": True,
                                    "displaylogo": False,
                                    "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                                    "responsive": True,
                                },
                            ),
                        ],
                    )
                ],
            ),
        ],
        style={
            "padding": "2rem clamp(1rem, 4vw, 3rem)",
            "maxWidth": "1400px",
            "margin": "0 auto",
            "minHeight": "calc(100vh - 8vh - 6vh)",  # Account for header and footer
        },
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

    # Define colors for better visual distinction
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Add all data for each model to the figure
    for idx, (model_name, model) in enumerate(existing_models.items()):
        data = [
            model.testing_RMSE,
            model.testing_R_squared,
            model.testing_MAE,
            model.testing_2fold_error,
            model.testing_RMSE,
        ]

        model_type = model.model_name
        model_label = f"{model_name} - {model_type}"

        # Use different colors for each model
        color = colors[idx % len(colors)]

        figure.add_trace(
            go.Scatterpolar(
                r=data,
                theta=categories,
                name=model_label,
                line=dict(color=color, width=3),
                marker=dict(color=color, size=8),
                fill="tonext" if idx > 0 else "toself",
                fillcolor=(
                    f"rgba{tuple(list(bytes.fromhex(color[1:])) + [0.1])}"
                    if color.startswith("#")
                    else color
                ),
            )
        )

    # Enhanced layout with better responsiveness and larger size
    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1],
                tickfont=dict(size=16, color="#333"),
                gridcolor="#ddd",
                linecolor="#666",
            ),
            angularaxis=dict(
                tickfont=dict(size=18, color="#333", family="Arial, sans-serif"),
                linecolor="#666",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=14, color="#333"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
        font=dict(family="Arial, sans-serif", color="#333", size=14),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=80, t=80, b=120),
        # Force a larger size
        autosize=True,
        width=None,
        height=700,  # Set a fixed minimum height
    )

    return figure
