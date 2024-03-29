import globals

import plotly.graph_objects as go

from dash import html, dcc, callback, Input, Output


def create_layout():
    """
    This function creates the main graph for the testing output statistics.
    """

    return html.Div(
        id='output-statistics-page',
        children=[
            html.Div(
                id='statistics-page-header',
                children=[
                    html.H1(
                        children=["Testing output statistics graph"]
                    )
                ]
            ),
            html.Div(
                id='statistics-page-contents',
                children=[
                    dcc.Graph(
                        id='statistics-graph',
                        style={
                            'height': '600px'
                        }
                    )
                ]
            )
        ]
    )


@callback(
    Output('statistics-graph', 'figure'),
    Input('statistics-graph', 'id')
)
def update_statistics_graph(_id):
    """
    This callback generates the output statistics graph based on the existing models
    """

    # only stores the created models (in case the user adds more models but doesn't initialise all of them)
    # to avoid errors when plotting the graph
    existing_models = {}
    for key, value in globals.MODELS_LIST.items():
        if value is not None:
            existing_models[key] = value

    figure = go.Figure()

    categories = ['RMSE', 'R-squared', 'MAE', 'Two-fold error', 'RMSE']

    # Add all data for each model to the figure
    for model_name, model in existing_models.items():
        data = [model.testing_RMSE, model.testing_R_squared, model.testing_MAE,
                model.testing_2fold_error, model.testing_RMSE]

        figure.add_trace(go.Scatterpolar(
            r=data,
            theta=categories,
            name=model_name
        ))

    figure.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )),
        showlegend=True
    )

    return figure
