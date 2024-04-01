# *ezSTEP*: an online platform for training and testing machine learning models for protein expression

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/logo.png)

## Introduction

*ezSTEP* (Sequence-to-expression Predictor) is a machine-learning platform with a web-based user interface that enables the construction of automated machine-learning pipelines for computational analysis and predictions using regression.

## Accessibility

*ezSTEP* can be accessed in two different ways:

- our web app is hosted on the following server, which users can access using the following link:

  - https://ezstep-f617792399bb.herokuapp.com (Note: this link will become deprecated in the future)

  - http://calbuco.inf.ed.ac.uk/ezSTEP (Note: if the link didn't work because it became deprecated, our platform can be accessed through this link instead)

- alternatively, users can install the app on their own local machine, and run it there (see the instructions below)

## Methods

Below you can see an outline of the pipeline used in our app. These are all the steps that our platform performs based on the user's inputs, and the underlying computations are abstracted from the user to ensure a nice and user-friendly experience.

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/pipeline.png)

## Repository structure

Below we provide a breakdown of the most important folders in this repository, and what they each contain:

  - ```.github/workflows```: this contains the continuous integration (CI) pipeline for the app, ensuring that no malfunctioning code is deployed to the server.

  - ```app```: this contains all the code for the app currently hosted on the server; this version differs from ```local_version``` due to changes that were needed in order to ensure the server can handle a multi-user enviornment in a safe and secure manner, which is why Redis was used for storing data.

  - ```datasets_used```: this contains all the datasets used for training and testing the app, as well as the Python scripts used for obtaining those datasets (NOTE: since the splits were done with random sampling, re-running the scripts may result in different datasets). The links for accessing each of the three original datasets used can be found below:

      - https://github.com/JeschekLab/uASPIre/tree/master/RBS_data (for the RBS data from the Hollerer et al. (2020) paper, the file is called uASPIre_RBS_300k_r2.txt)
        
      - https://www.nature.com/articles/nbt.4238#MOESM44 (for the coding sequences data from the Cambray et al. (2018) paper, the file is called Ecoli_data.csv)
        
      - https://www.nature.com/articles/s41586-022-04506-6 (for the promoter sequences data from the Vaishnav et al. (2022) paper, the file is called yeast_data.csv)

  - ```downloadable_data```: this contains the example datasets for training, testing and querying data, which can be downloaded directly from our platform. These datasets were also obtained from the coding sequences data from the Cambray et al. (2018) paper, again using a random split.

  - ```local_version```: this contains all the code for the version of the app that can be installed and set up on a local device. The main difference is that, unlike the server version, there is no need to ensure a mult-user environment using Redis, and so global variables stored on the local device are used for handling the app's data instead.

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

## Citations

If you found *ezSTEP* useful, please kindly cite the following paper:

Andreas Hiropedi, Yuxin Shen, Diego A. Oyarzún, *ezSTEP*: a web-based tool for Sequence-To-Expression Prediction, 2024.
