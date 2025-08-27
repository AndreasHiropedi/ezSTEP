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
        className="responsive-guidelines-container",
        children=[
            html.P(
                "Below are some guidelines and information about how the platform works. It includes some "
                "general information about the structure and layout of the app, as well as some more "
                "specific information about the individual tools available.",
                className="responsive-guidelines-intro",
            ),
            html.Div(
                id="info-wrapper",
                className="responsive-info-wrapper",
                children=[
                    # General information
                    html.Div(
                        id="general-info",
                        className="responsive-info-card general-info-card",
                        children=[
                            html.H4(
                                "1. General information",
                                className="responsive-card-title",
                            ),
                            html.Hr(className="responsive-card-divider"),
                            html.P(
                                "The app consists of three sections: File upload, Model input parameters, and Model outputs. "
                                "In order for the user to see the model output, their inputted parameters for the selected model, "
                                "as well as their uploaded dataset, must be first validated and then processed. Once these steps "
                                "have occurred, the user will be able to visualise the model output (see more in the 'Model outputs' section). "
                                "For more detailed information on each specific subsection, see the information below.",
                                className="responsive-card-text",
                            ),
                        ],
                    ),
                    # File upload
                    html.Div(
                        id="file-upload",
                        className="responsive-info-card file-upload-card",
                        children=[
                            html.H4(
                                "2. File upload", className="responsive-card-title"
                            ),
                            html.Hr(className="responsive-card-divider"),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_upload_image}",
                                className="responsive-guidelines-image",
                            ),
                            html.P(
                                "As can be seen in the image above, this section contains three upload boxes. "
                                "The required fields are for uploading the training and testing data (in order to train the selected model), "
                                "and the optional field is for uploading a dataset for querying the model on unlabelled data. "
                                "For each of the three fields, the user has a choice of how they wish to upload the data: "
                                "they can either upload a file, or paste their data in a textbox.",
                                className="responsive-card-text",
                            ),
                            html.P(
                                "If they choose to upload a file, they must ensure the file contains at least one column with all the sequence data, "
                                "and one column with all the labels information, with these two columns being matched. If they choose to use the textbox, "
                                "they must ensure the data is formatted in the following order: sequence + separator (such as , or | or ;) + label + new line character.",
                                className="responsive-card-text",
                            ),
                            html.P(
                                "NOTE: for the training process, we always perform 5-fold cross validation on the uploaded training dataset. "
                                "Therefore, if the length of the text input is less than 5 sequences long, even if the sequences are valid, "
                                "the input will be rendered invalid.",
                                className="responsive-card-text card-note",
                            ),
                        ],
                    ),
                    # Model input parameters
                    html.Div(
                        id="model-input",
                        className="responsive-info-card model-input-card",
                        children=[
                            html.H4(
                                "3. Model input parameters",
                                className="responsive-card-title",
                            ),
                            html.Hr(className="responsive-card-divider"),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_input_image}",
                                className="responsive-guidelines-image",
                            ),
                            html.P(
                                "In this section, the user gets to select a model and input all the necessary information "
                                "in order to train and test that model. This information will be used together with the datasets uploaded "
                                "(see the previous section for more details) in order to train the models and visualise the output.",
                                className="responsive-card-text",
                            ),
                            html.P(
                                "The user will see one (or more) hyperlink(s) depending on the number of models they have added. "
                                "In order to input all the necessary information, the user will need to click on these hyperlinks individually, "
                                "which will prompt them to a new page where they can input all the data for a specific model.",
                                className="responsive-card-text",
                            ),
                        ],
                    ),
                    # Model output visualisations
                    html.Div(
                        id="model-output",
                        className="responsive-info-card model-output-card",
                        children=[
                            html.H4(
                                "4. Model outputs", className="responsive-card-title"
                            ),
                            html.Hr(className="responsive-card-divider"),
                            html.Img(
                                src=f"data:image/png;base64,{encoded_output_image}",
                                className="responsive-guidelines-image",
                            ),
                            html.P(
                                "Once the data has been uploaded and the user has set all the input parameters, "
                                "the visualisations for the specific model, along with a spider plot showing several output statistics, are generated. "
                                "The four main summary statistics used are: Root Mean Squared Error (RMSE), R-squared, "
                                "Mean Absolute Error (MAE), and Percentage (%) within 2-fold error.",
                                className="responsive-card-text",
                            ),
                            html.P(
                                "Similarly to model inputs, the user will see one (or more) hyperlink(s) depending on "
                                "the number of models they have added. On each of these pages, there will be several graphs displayed, "
                                "along with a summary of the user's selected inputs and the model's performance. All plots are interactive, "
                                "with features such as zooming, saving as PNG, and focusing on specific regions.",
                                className="responsive-card-text",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
