# Heart Condition Checker
The app created with Python to predict person's heart health condition based on well-trained machine learning model (logistic regression).

![App overview](https://i.imgur.com/4wTlvKj.png)

## Table of Contents
1. [General info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)


## General info
In this project, logistic regression was used to predict person's heart health condition expressed as a dichotomous variable (heart disease: yes/no). The model was trained on approximately 70,000 data from an annual telephone survey of the health of U.S. residents from the year 2020. The dataset is publicly available at the following link: https://www.cdc.gov/brfss/annual_data/annual_2020.html. The data is originally stored in SAS format. The original dataset contains approx. 400,000 rows and over 200 variables. The data conversion and cleaning process is described in another repository: https://github.com/kamilpytlak/data-analyses/tree/main/heart-disease-prediction. This project contains:
* the app - the application construct is located in the `app.py` file. This file uses data from the `data` folder and saved (previously trained) ML models from the `model` folder.

The logistic regression model was found to be satisfactorily accurate (accuracy approx. 80%).

## Technologies
The app is fully written in Python 3.9.9. `streamlit 1.5.1` was used to create the user interface, and the machine learning itself was designed using the module `scikit-learn 1.0.2`. `pandas 1.41.`, `numpy 1.22.2` and `polars 0.13.0` were used to perform data converting operations.

## Installation
The project was uploaded to the web using heroku. You can use it online at the following link: https://share.streamlit.io/kamilpytlak/heart-condition-checker/main/app.py. If you want to use this app on your local machine, make sure that you have installed the necessary modules in a version no smaller than the one specified in the `requirements.txt` file. You can either install them globally on your machine or create a virtual environment (`pipenv`), which is highly recommended.
1.  Install the packages according to the configuration file `requirements.txt`.
```
pip install -r requirements.txt
```

2.  Ensure that the `streamlit` package was installed successfully. To test it, run the following command:
```
streamlit hello
```
If the example application was launched in the browser tab, everything went well. You can also specify a port if the default doesn't respond:
```
streamlit hello --server.port port_number
```
Where `port_number` is a port number (8889, for example).

3.  To start the app, type:
```
streamlit run app.py
```

And that's it! Now you can predict your heart health condition expressed as a binary variable based on a dozen factors that best describe you.
