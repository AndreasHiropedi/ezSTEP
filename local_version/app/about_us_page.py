import base64

from dash import html


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
            about_us(),
            useful_links(),
            app_footer()
        ]
    )


def app_footer():
    """
    This function builds the footer for the web app.
    """

    return html.Footer(
        id='app-footer',
        children=[
            # University logo
            html.A(
                children=[
                    html.Img(
                        src='data:image/png;base64,{}'.format(
                            base64.b64encode(
                                open(
                                    './assets/eduni-logo.png', 'rb'
                                ).read()
                            ).decode()
                        ),
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


def useful_links():
    """
    This function builds the container for the links to the user guidelines and the disclaimer
    pages, as well as explanations for what can be found on those pages.
    """

    return html.Div(
        id='useful-links',
        children=[
            html.H2(
                'Useful links',
                style={
                    'margin-left': '40px',
                    'font-size': '20pt'
                }
            ),
            html.A(
                children=[
                    html.H4(
                        "1. User guidelines",
                        id='guidelines-ref',
                        style={
                            'margin-left': '60px',
                            'font-size': '16pt'
                        }
                    )
                ],
                href='/user-guidelines/',
                target='_blank'
            ),
            user_guidelines_explanation(),
            html.A(
                children=[
                    html.H4(
                        "2. Disclaimer",
                        id='disclaimer-ref',
                        style={
                            'margin-left': '60px',
                            'font-size': '16pt'
                        }
                    )
                ],
                href='/disclaimer/',
                target='_blank'
            ),
            disclaimer_explanation()
        ]
    )


def user_guidelines_explanation():
    """
    This function builds the body of text explaining what the user guidelines page contains.
    """

    return html.Div(
        id='guidelines-about-us-section',
        children=[
            html.P(
                "This first link redirects the user to a separate page containing a series of guidelines on the "
                "features available as part of our platform. These guidelines should provide the users with enough "
                "information to easily find their way around the platform, as well as clarify any questions users "
                "may have.",
                style={
                    'margin-left': '60px',
                    'margin-right': '40px',
                    'font-size': '12pt',
                    'margin-bottom': '20px'
                }
            )
        ]
    )


def disclaimer_explanation():
    """
    This function builds the body of text explaining what the disclaimer page contains.
    """

    return html.Div(
        id='disclaimer-about-us-section',
        children=[
            html.P(
                "This second link redirects the user to a disclaimer page, which we have created to show that "
                "we only retain information provided by the users on our servers whilst they are using the app, and "
                "that information is deleted once they close the app in their browser. ",
                style={
                    'margin-left': '60px',
                    'margin-right': '40px',
                    'font-size': '12pt',
                    'margin-bottom': '20px'
                }
            )
        ]
    )


def about_us():
    """
    This function builds the 'Who we are' body of text.
    """

    return html.Div(
        id='about-us-section',
        children=[
            html.H2(
                'Who we are',
                style={
                    'margin-left': '40px',
                    'font-size': '20pt',
                    'margin-top': '20px'
                }
            ),
            html.P(
                "The Biomolecular control group is a research group located at the School of Informatics and the "
                "School of Biological Sciences of the University of Edinburgh, in one of the most vibrant "
                "European capitals. We are members of the Edinburgh Centre for Engineering Biology. "
                "Our team develops computational methods to study molecular processes in living cells. "
                "We use mathematics to understand natural networks and to design novel systems for biotechnology "
                "and biomedicine. We employ a wide range of methods, such as machine learning, control theory, "
                "stochastic analysis and network theory. The group lead is Diego Oyarzún and includes research "
                "students and postdocs. ",
                style={
                    'margin-left': '40px',
                    'margin-right': '40px',
                    'font-size': '12pt',
                    'margin-bottom': '20px'
                }
            )
        ],
        style={
            'margin-top': '30px'
        }
    )
