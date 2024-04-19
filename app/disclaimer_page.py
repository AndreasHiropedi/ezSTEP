import base64
import os

from dash import html


def create_layout():
    """
    This function creates the layout for the 'About us' page.
    """

    return html.Div(
        children=[
            html.H1(
                'Disclaimer',
                id='disclaimer-header'
            ),
            disclaimer_text(),
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


def disclaimer_text():
    """
    This function generates the main body of text to be displayed on the disclaimer page.
    """

    return html.Div(
        id='disclaimer-text',
        children=[
            html.H2(
                'Privacy Disclaimer',
                style={
                    'margin-left': '40px',
                    'font-size': '20pt'
                }
            ),
            html.P(
                "Please be advised that this web-app does not store or retain any personal data or information "
                "entered during your session. All data is deleted and cannot be recovered once you close "
                "this browser tab. We respect your privacy and ensure that your session remains confidential.",
                style={
                    'margin-top': '40px',
                    'margin-left': '40px',
                    'margin-right': '40px',
                    'font-size': '12pt',
                    'margin-bottom': '20px'
                }
            ),
            html.H2(
                'Dataset Disclaimer',
                style={
                    'margin-left': '40px',
                    'font-size': '20pt'
                }
            ),
            html.P(
                "The example datasets provided within this platform are obtained from publicly available sources. "
                "These datasets are intended for demonstration and educational purposes only, and were obtained from "
                "the Cambray et al. (2018) paper.",
                style={
                    'margin-top': '40px',
                    'margin-left': '40px',
                    'margin-right': '40px',
                    'font-size': '12pt',
                    'margin-bottom': '20px'
                }
            )
        ]
    )
