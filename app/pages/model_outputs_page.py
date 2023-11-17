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
                    output_graph_placeholder_card(model_count)
                ]
            )
        ]
    )


def output_graph_placeholder_card(model_count):
    """
    This function creates the card with the placeholder output graph
    """

    return dbc.Card(
        id='card-output',
        children=[
            dbc.CardHeader(
                id='card-header-output',
                children=["Interactive scatter plot with Iris dataset"]
            ),
            dbc.CardBody(
                id='card-body-output',
                children=[
                    # Placeholder
                    dcc.Graph(id={'type': 'scatter-plot', 'index': model_count}),
                    html.P("Filter by petal width:"),
                    dcc.RangeSlider(
                        id={'type': 'range-slider', 'index': model_count},
                        min=0, max=2.5, step=0.1,
                        marks={0: '0', 2.5: '2.5'},
                        value=[0.5, 2]
                    )
                ]
            )
        ]
    )


@callback(
    Output({'type': 'scatter-plot', 'index': MATCH}, 'figure'),
    [Input({'type': 'range-slider', 'index': MATCH}, 'value'),
     Input('store-model-count', 'data')])
def update_placeholder(slider_range, model_data):
    """
    This callback function keeps track of the user changes to the
    displayed graph (as part of the output information cards)
    """

    _model_count = model_data['n_clicks']

    df = px.data.iris()  # replace with your own data source
    low, high = slider_range
    mask = (df['petal_width'] > low) & (df['petal_width'] < high)
    fig = px.scatter(
        df[mask], x="sepal_width", y="sepal_length",
        color="species", size='petal_length',
        hover_data=['petal_width'])

    return fig

