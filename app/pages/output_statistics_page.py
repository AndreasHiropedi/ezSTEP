import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import html, dcc, callback, Input, Output

# Sample data (replace with actual data)
df = pd.DataFrame({
    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Grapes', 'Pineapples'],
    'Amount': [4, 1, 2, 2, 5],
})

# Generate a color map dynamically based on the number of fruits
# Using a Plotly qualitative color palette
colors = px.colors.qualitative.Plotly
color_map = {fruit: colors[i % len(colors)] for i, fruit in enumerate(df['Fruit'])}  # replace with actual data


def create_layout(metric_name):
    """
    This function creates the pages for the output statistics graphs.
    """

    if metric_name == 'RMSE':
        return html.Div(
            id='RMSE-page',
            children=[
                html.Div(
                    id='rmse-page-header',
                    children=[
                        html.H1(
                            children=["RMSE bar chart"]
                        )
                    ]
                ),
                html.Div(
                    id='rmse-page-contents',
                    children=[
                        dcc.Graph(
                            id='rmse-bar-chart',
                            style={
                                'height': '600px'
                            }
                        )
                    ]
                )
            ]
        )

    elif metric_name == 'R-squared':
        return html.Div(
            id='R-squared-page',
            children=[
                html.Div(
                    id='r-squared-page-header',
                    children=[
                        html.H1(
                            children=["R-squared bar chart"]
                        )
                    ]
                ),
                html.Div(
                    id='r-squared-page-contents',
                    children=[
                        dcc.Graph(
                            id='r-squared-bar-chart',
                            style={
                                'height': '600px'
                            }
                        )
                    ]
                )
            ]
        )

    elif metric_name == 'MAE':
        return html.Div(
            id='MAE-page',
            children=[
                html.Div(
                    id='mae-page-header',
                    children=[
                        html.H1(
                            children=["MAE bar chart"]
                        )
                    ]
                ),
                html.Div(
                    id='mae-page-contents',
                    children=[
                        dcc.Graph(
                            id='mae-bar-chart',
                            style={
                                'height': '600px'
                            }
                        )
                    ]
                )
            ]
        )

    elif metric_name == 'Percentage within 2-fold error':
        return html.Div(
            id='two-fold-error-page',
            children=[
                html.Div(
                    id='two-fold-page-header',
                    children=[
                        html.H1(
                            children=["Percentage within 2-fold error bar chart"]
                        )
                    ]
                ),
                html.Div(
                    id='two-fold-page-contents',
                    children=[
                        dcc.Graph(
                            id='fold-error-bar-chart',
                            style={
                                'height': '600px'
                            }
                        )
                    ]
                )
            ]
        )

    return 'No content available'


@callback(
    Output('rmse-bar-chart', 'figure'),
    Input('rmse-bar-chart', 'id')
)
def update_rmse_bar_chart(_id):
    """
    This callback generates the RMSE bar chart based on the existing models
    """

    # Create an empty figure
    fig = go.Figure()

    # Add a bar for each category
    for fruit in df['Fruit']:
        fig.add_trace(go.Bar(
            x=[fruit],
            y=[df[df['Fruit'] == fruit]['Amount'].values[0]],
            name=fruit,
            marker_color=color_map[fruit]
        ))

    fig.update_layout(
        showlegend=False,
        xaxis_title='Model number',
        yaxis_title='RMSE test value'
    )

    return fig


@callback(
    Output('r-squared-bar-chart', 'figure'),
    Input('r-squared-bar-chart', 'id')
)
def update_r_squared_bar_chart(_id):
    """
    This callback generates the R-squared bar chart based on the existing models
    """

    # Create an empty figure
    fig = go.Figure()

    # Add a bar for each category
    for fruit in df['Fruit']:
        fig.add_trace(go.Bar(
            x=[fruit],
            y=[df[df['Fruit'] == fruit]['Amount'].values[0]],
            name=fruit,
            marker_color=color_map[fruit]
        ))

    fig.update_layout(
        showlegend=False,
        xaxis_title='Model number',
        yaxis_title='R-squared test value'
    )

    return fig


@callback(
    Output('mae-bar-chart', 'figure'),
    Input('mae-bar-chart', 'id')
)
def update_mae_bar_chart(_id):
    """
    This callback generates the MAE bar chart based on the existing models
    """

    # Create an empty figure
    fig = go.Figure()

    # Add a bar for each category
    for fruit in df['Fruit']:
        fig.add_trace(go.Bar(
            x=[fruit],
            y=[df[df['Fruit'] == fruit]['Amount'].values[0]],
            name=fruit,
            marker_color=color_map[fruit]
        ))

    fig.update_layout(
        showlegend=False,
        xaxis_title='Model number',
        yaxis_title='MAE test value'
    )

    return fig


@callback(
    Output('fold-error-bar-chart', 'figure'),
    Input('fold-error-bar-chart', 'id')
)
def update_two_fold_error_bar_chart(_id):
    """
    This callback generates the Percentage 2-fold error bar chart based on the existing models
    """

    # Create an empty figure
    fig = go.Figure()

    # Add a bar for each category
    for fruit in df['Fruit']:
        fig.add_trace(go.Bar(
            x=[fruit],
            y=[df[df['Fruit'] == fruit]['Amount'].values[0]],
            name=fruit,
            marker_color=color_map[fruit]
        ))

    fig.update_layout(
        showlegend=False,
        xaxis_title='Model number',
        yaxis_title='Percentage two-fold error for testing'
    )

    return fig
