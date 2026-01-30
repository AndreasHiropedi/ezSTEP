# *ezSTEP*: an online platform for training and testing machine learning models for protein expression

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/logo.png)

## Introduction

*ezSTEP* (Sequence-to-expression Predictor) is a machine-learning platform with a web-based user interface that enables the construction of automated machine-learning pipelines for computational analysis and predictions using regression.

## Methods

Below you can see an outline of the pipeline used in our app. These are all the steps that our platform performs based on the user's inputs, and the underlying computations are abstracted from the user to ensure a nice and user-friendly experience.

![ezSTEP](https://github.com/AndreasHiropedi/ezSTEP/blob/main/pipeline.png)

## Repository structure

Below we provide a breakdown of the most important files and folders in this repository, and what they each contain:

  - ```.github/workflows```: this contains the continuous integration (CI) pipeline for the app, ensuring that no malfunctioning code is deployed to the server.

  - ```main.py```: this contains the code for the main entry point for the app, as well as for the layout of the home page, which is the first page the user sees when launching the app

  - ```config.py```: this is a configuration file used to set up the environment for launching the app, both locally and on the nihito platform

  - ```.env```: this contains the actual environment configurations

  - ```assets```: this contains all the frontend-relevant files, such as the customized CSS styling and the additional JS scripts to make the platform more dynamic and interactive

  - ```data_processing```: this contains all the datasets used for training and testing the app, as well as the Python scripts used for obtaining those datasets (NOTE: since the splits were done with random sampling, re-running the scripts may result in different datasets). The links for accessing each of the three original datasets used can be found below:

      - https://github.com/JeschekLab/uASPIre/tree/master/RBS_data (for the RBS data from the Hollerer et al. (2020) paper, the file is called uASPIre_RBS_300k_r2.txt)
        
      - https://www.nature.com/articles/nbt.4238#MOESM44 (for the coding sequences data from the Cambray et al. (2018) paper, the file is called Ecoli_data.csv)
        
      - https://www.nature.com/articles/s41586-022-04506-6 (for the promoter sequences data from the Vaishnav et al. (2022) paper, the file is called yeast_data.csv)

  - ```database```: this contains all the code to set up, populate, and clean the SQLite database used for storing the user provided data (NOTE: the data is stored temporarily and the database is cleaned periodically in order to
  remove data from expired sessions)

  - ```models```: this contains all the code for the different models that we make available on the platform, and includes the code on training, testing, evaluating and performing hyperparameter optimisation. The four models available on the platform are:

    - **Random Forest**

    - **Ridge Regressor**

    - **Multi-layer Perceptron**

    - **Support Vector Machine**

  - ```pages```: since ezSTEP is a multi-page web-app, this folder contains all the code for all the other pages available on the platform, such as the 'About Us' page, the user guidelines, and the pages allowing users to create models and view their results

  - ```utils```: this contains all the code providing the model customization functionality offered on the platform, including feature encoding, feature normalization, feature selection and performing unsupervised learning on the data

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

6) Once all dependencies are installed, the app can be run with the following command:

```sh
python main.py
```

This command will create a localhost link, which, when clicked, will redirect you to your local instance of the app.

**Optional:** since the code creates a ```.db``` file locally, the data from one session to another may persist. To avoid this, run these commands in order to start each iteration with a clean slate:

```sh
rm ezstep.db
python main.py
```

## Citations

If you found *ezSTEP* useful, please kindly cite the following paper:

Andreas Hiropedi, Yuxin Shen, Diego A. Oyarzún, *ezSTEP: a no-code tool for sequence-to-expression prediction
in synthetic biology*, 2025 (in proceedings).
