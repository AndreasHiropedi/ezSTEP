import base64
import os

from dash import html


def create_layout():
    """
    This function creates the layout for the 'User guidelines' page.
    """

    return html.Div(
        children=[
            html.H1("User Guidelines", id="guidelines-header"),
            user_guide(),
            app_footer(),
        ]
    )


def app_footer():
    """
    This function builds the footer for the web app.
    """

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Go one level up to the root folder
    root_directory = os.path.abspath(os.path.join(current_directory, ".."))

    # Construct the absolute path to the image file
    uni_image_path = os.path.join(root_directory, "assets", "eduni-logo.png")

    # Open the image, read it, and encode it into Base64
    encoded_uni_image = base64.b64encode(open(uni_image_path, "rb").read()).decode()

    return html.Footer(
        id="app-footer",
        children=[
            # University logo
            html.A(
                children=[html.Img(src=f"data:image/png;base64,{encoded_uni_image}")],
                href="https://homepages.inf.ed.ac.uk/doyarzun/",
                target="_blank",
            ),
            # Copyright
            html.H3(
                "Biomolecular Control Group 2024",
            ),
        ],
        style={"background": "white", "color": "black"},
    )


def user_guide():
    """
    This function builds the user guidelines section of the web app,
    which provides detailed information about how the user can interact
    with the platform.
    """

    # Determine the absolute path of the current file (e.g., main_page.py)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Go one level up to the root folder
    root_directory = os.path.abspath(os.path.join(current_directory, ".."))

    # Construct the absolute path to the image file
    upload_image_path = os.path.join(root_directory, "assets", "upload.png")

    # Open the image, read it, and encode it into Base64
    encoded_upload_image = base64.b64encode(
        open(upload_image_path, "rb").read()
    ).decode()

    # Construct the absolute path to the image file
    input_image_path = os.path.join(root_directory, "assets", "inputs.png")

    # Open the image, read it, and encode it into Base64
    encoded_input_image = base64.b64encode(open(input_image_path, "rb").read()).decode()

    # Construct the absolute path to the image file
    output_image_path = os.path.join(root_directory, "assets", "outputs.png")

    # Open the image, read it, and encode it into Base64
    encoded_output_image = base64.b64encode(
        open(output_image_path, "rb").read()
    ).decode()

    return html.Div(
        id="user-guide",
        children=[
            html.H1("User Guidelines"),
            html.P(
                "Below are some guidelines and information about how the platform works. It includes some "
                "general information about the structure and layout of the app, as well as some more "
                "specific information about the individual tools available. "
            ),
            html.Div(
                id="info-wrapper",
                children=[
                    # General information
                    html.Div(
                        id="general-info",
                        children=[
                            html.H4("1. General information"),
                            html.Hr(),
                            html.P(
                                children=[
                                    "The app consists of three sections: ",
                                    html.Br(),
                                    html.Br(),
                                    "   1. File upload",
                                    html.Br(),
                                    html.Br(),
                                    "   2. Model input parameters",
                                    html.Br(),
                                    html.Br(),
                                    "   3. Model outputs",
                                    html.Br(),
                                    html.Br(),
                                    "In order for the user to see the model output, their inputted "
                                    "parameters for the selected model, as well as their uploaded "
                                    "dataset, must be first validated and then processed. Once these "
                                    "steps have occurred, the user will be able to visualise the model "
                                    "output (see more in the 'Model outputs' section). For "
                                    "more detailed information on each specific subsection, see the "
                                    "information below. ",
                                ]
                            ),
                        ],
                        style={
                            "background": "#f8d7da",
                            "color": "#721c24",
                            "borderColor": "#f5c6cb",
                        },
                    ),
                    # File upload
                    html.Div(
                        id="file-upload",
                        children=[
                            html.H4("2. File upload"),
                            html.Hr(),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_upload_image}"
                            ),
                            html.P(
                                "As can be seen in the image above, this section contains three upload boxes. "
                                "The required fields are for uploading "
                                "the training and testing data (in order to train the selected model), and the "
                                "optional field is for uploading a dataset for querying the model on unlabelled data. "
                                "For each of the three fields, the user has a choice of how they wish to "
                                "upload the data: they can either upload a file, or paste their data in a "
                                "textbox. If they choose to upload a file, they must ensure the file "
                                "contains at least one column with all the sequence data, and one column "
                                "with all the labels information, with these two columns being matched. If "
                                "they choose to use the textbox, they must ensure the data is formatted in "
                                "the following order: sequence + separator (such as , or | or ;) + label + "
                                "new line character. If the user fails to ensure these conditions, then "
                                "the app will not be able to process their inputted data and will inform the user. "
                                "However, it is worth noting "
                                "that these conditions only apply for training and testing, since for the querying "
                                "data the user will only need to provide the sequences (in the format of one sequence "
                                "per line)."
                            ),
                            html.P(
                                "NOTE: for the training process, we always perform 5-fold cross validation on the "
                                "uploaded training dataset. Therefore, it is worth noting that, if the length of the "
                                "text input is less than 5 sequences long, even if the sequences are valid, the input "
                                "will be rendered invalid. "
                            ),
                        ],
                        style={
                            "background": "#e2e3e5",
                            "color": "#383d41",
                            "borderColor": "#d6d8db",
                        },
                    ),
                    # Model input parameters
                    html.Div(
                        id="model-input",
                        children=[
                            html.H4("3. Model input parameters"),
                            html.Hr(),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_input_image}"
                            ),
                            html.P(
                                "In this section, the user gets to select a model and input all the "
                                "necessary information in order to train and test that model. This "
                                "information will be used together with the datasets uploaded (see the previous "
                                "section for more details) in order to train the models and visualise the "
                                "output (see the 'Model Outputs' section for more). The user "
                                "will also be able to add more than one model, but will need to input all "
                                "the necessary information for each model added. There will also be an  "
                                "option that will allow the user to choose whether or not they wish to  "
                                "optimise the model's hyperparameters or not."
                            ),
                            html.P(
                                "The user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added (see image). "
                                "In order to input all the necessary "
                                "information, the user will need to click on these hyperlinks individually, "
                                "which will prompt them to a new page where they can input all the data for a "
                                "specific model (see image). More information about the specifics of "
                                "model inputs can be found "
                                "in the user guidelines on the individual model input pages."
                            ),
                        ],
                        style={
                            "background": "#cce5ff",
                            "color": "#004085",
                            "borderColor": "#b8daff",
                        },
                    ),
                    # Model output visualisations
                    html.Div(
                        id="model-output",
                        children=[
                            html.H4("4. Model outputs"),
                            html.Hr(),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_output_image}"
                            ),
                            html.P(
                                children=[
                                    "Once the data has been uploaded and the user has set all the input parameters, "
                                    "the visualisations for the specific model, along with a spider plot showing "
                                    "several output statistics, are generated. These statistics are calculated for "
                                    "each model, and these are all displayed in the spider plot for each model "
                                    "(see image). "
                                    "The four main summary statistics used are: "
                                    "Root Mean Squared Error (RMSE), "
                                    "R-squared, "
                                    "Mean Absolute Error (MAE), "
                                    "and Percentage (%) within 2-fold error."
                                ]
                            ),
                            html.P(
                                "Similarly to model inputs, the user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added (see image). "
                                "On each of these pages, there will be several "
                                "graphs displayed (such as a scatter plot of predicted versus actual values, or bar "
                                "charts showing the feature correlation to the target variable (protein)), along with "
                                "a summary of the user's selected inputs and the model's performance in training and "
                                "testing. Depending on the additional inputs the user provides (such as query data, or "
                                "enabling feature selection), additional graphs will be generated and displayed on "
                                "these individual pages."
                                "All plots available in this app are interactive, "
                                "and using the legend on the side, the user can select "
                                "which models they wish to have displayed in the graph, and they can simply "
                                "enable/ disable them by clicking on the respective item in the legend. Additionally, "
                                "all graphs provide features such as zooming in/ out, saving the graph as a PNG to "
                                "the user's local device, and selecting to focus only on certain regions of the graph."
                            ),
                        ],
                        style={
                            "background": "#fff3cd",
                            "color": "#856404",
                            "borderColor": "#ffeeba",
                        },
                    ),
                ],
            ),
        ],
    )
