# ezSTEP: an online platform for training and testing machine learning models for protein expression

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/logo.png)

## Introduction
ezSTEP (Sequence-to-expression Predictor) is the first machine-learning platform with a web-based user interface that enables the construction of automated machine-learning pipelines for computational analysis and predictions using regression.

## Methods
Below you can see an outline of the pipeline used in our app. These are all the steps that our platform performs based on the user's inputs, and the underlying computations are abstracted from the user to ensure a nice and user-friendly experience.

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/pipeline.png)

## Accessibility
ezSTEP can be accessed in two different ways:

- our web app is hosted on the following server, which users can access using the following link:

  *insert link*

- alternatively, users can install the app on their own local machine, and run it there (see the instructions below)

## Installation

To install the app and run it locally, the users can follow the procedure outlined below:

1) Ensure Python is installed on your device. For advice on how to do that, see https://www.python.org

2) Install all the necessary dependencies. This can be done by installing everything contained within the requirements.txt file using the following command:

```sh
pip install -r requirements. txt
```

3) Once all dependencies are installed, the app can be launched by running the following command:

```sh
python app/main_page.py
```

This command will create a localhost link, which, when clicked, will redirect you to your local instance of the app.

