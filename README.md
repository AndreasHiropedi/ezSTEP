# ezSTEP: an online platform for training and testing machine learning models for protein expression

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/logo.png)

## Introduction

ezSTEP (Sequence-to-expression Predictor) is the first machine-learning platform with a web-based user interface that enables the construction of automated machine-learning pipelines for computational analysis and predictions using regression.

## Accessibility

ezSTEP can be accessed in two different ways:

- our web app is hosted on the following server, which users can access using the following link:

  https://ezstep-f617792399bb.herokuapp.com

- alternatively, users can install the app on their own local machine, and run it there (see the instructions below)

## Methods

Below you can see an outline of the pipeline used in our app. These are all the steps that our platform performs based on the user's inputs, and the underlying computations are abstracted from the user to ensure a nice and user-friendly experience.

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/pipeline.png)

## Repository structure

Below we provide a breakdown of the most important folders in this repository, and what they each contain:

  - ```app```: this contains all the code for the app currently hosted on the server; this version differs from ```local_version``` due to changes that were needed in order to ensure the server can handle a multi-user enviornment in a safe and secure manner

  - ```datasets_used```: this contains all the datasets used for training and testing the app, as well as the Python scripts used for obtaining those datasets (NOTE: since the splits were done with random sampling, re-running the scripts may result in different datasets)

  - ```local_version```: this contains all the code for the version of the app that can be installed and set up on one's local device

## Installation

To install the app and run it locally, the users can follow the procedure outlined below:

1) Ensure Python is installed on your device. For advice on how to do that, see https://www.python.org (please ensure the Python version used is >= 3.8 and <= 3.11)

2) Ensure that you have Git configured on your device (see https://github.com/git-guides/install-git for details on how to install Git)

3) Once Python and git are set up, clone this repo using the following command:

```sh
git clone https://github.com/AndreasHiropedi/ezSTEP.git
```

4) After cloning the repository, change directory so that you are in the repository directory using the following command:

```sh
cd ezSTEP
```

5) Next, install all the necessary dependencies. This can be done by installing everything contained within the requirements.txt file using the following command:

```sh
pip install -r requirements.txt
```

6) Once all dependencies are installed, you will need to change directories to the local_version folder, and then the app folder using the following command:

```sh
cd local_version/app
```

7) Once in the app directory, the app can be run with the following command:

```sh
python main_page.py
```

This command will create a localhost link, which, when clicked, will redirect you to your local instance of the app.

