from dash import html


def create_layout(metric_name):
    """
    This function creates the pages for the output statistics graphs.
    """

    if metric_name == 'RMSE':
        return html.Div(
            id='RMSE-page',
            children=[
                # TODO: FILL THIS
            ]
        )

    elif metric_name == 'R-squared':
        return html.Div(
            id='R-squared-page',
            children=[
                # TODO: FILL THIS
            ]
        )

    elif metric_name == 'MAE':
        return html.Div(
            id='MAE-page',
            children=[
                # TODO: FILL THIS
            ]
        )

    elif metric_name == 'Percentage within 2-fold error':
        return html.Div(
            id='2-fold-error-page',
            children=[
                # TODO: FILL THIS
            ]
        )

    return 'No content available'

