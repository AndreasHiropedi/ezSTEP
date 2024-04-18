import base64
import os

from dash import html, dcc, callback, Input, Output


def create_layout():
    """
    This function creates the layout for the 'About us' page.
    """

    return html.Div(
        children=[
            html.H1(
                'About Us',
                id='about-us-header'
            ),
            # TODO: ADD BODY FOR THE PAGE
            app_footer()
        ]
    )


def app_footer():
    """
    This function builds the footer for the web app.
    """

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the image file
    uni_image_path = os.path.join(current_directory, 'assets', 'eduni-logo.png')
    # Open the image, read it, and encode it into Base64
    encoded_uni_image = base64.b64encode(open(uni_image_path, 'rb').read()).decode()

    return html.Footer(
        id='app-footer',
        children=[
            # University logo
            html.A(
                children=[
                    html.Img(
                        src=f'data:image/png;base64,{encoded_uni_image}'
                    )
                ],
                href='https://homepages.inf.ed.ac.uk/doyarzun/',
                target='_blank'
            ),

            # Copyright
            html.H3(
                "Biomolecular Control Group 2024",
            )
        ],
        style={
            'background': 'white',
            'color': 'black'
        }
    )
