# CelebrityRecogniser
 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [About](#About)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [Technologies Used](#technologies-used)
  * [Future scope of project](#future-scope)


## Demo
Link: https://share.streamlit.io/prakharsrivastavaa7/celebrityrecogniser/main/home.py


## Overview
This is a streamlit web app which inputs an image and recognises whether it matches any of the celebrity present in the dataset.

  ### Input Screenshots      

The landing page of the website takes user inputs for the image as shown below-

![image](https://user-images.githubusercontent.com/63156822/141937327-61237517-9257-4067-b0bf-9305f53ef6ab.png)


![image](https://user-images.githubusercontent.com/63156822/141937500-e056a272-a166-4e33-b3ea-f7bfe55578ef.png)



   ### Output Screenshots

After clicking on predict the output price will be shown as follows:

![image](https://user-images.githubusercontent.com/63156822/141937404-a6e7dd91-a02f-48b1-b6da-348eeca11d07.png)

Along with the name of the celebrity the webapp will also display the probability score for each of the data item as follows

![image](https://user-images.githubusercontent.com/63156822/141937423-a970296f-ee00-463d-94ce-b4f5757fd722.png)


  ### Flowchart

There are two main components of the project - Machine Learning based Model Fitting and Web Deployment Using Flask.
For model training the data is loaded and reduced in size to ensure proper deployment of herkou app. The data is preprocessed and unneccessary items are dropped from the data following which model training is done. We use XGB and RandomForestRegressor to traain the dataset. Hyperparameter tuning is also used in case of RandomForestRegressor to get the best parameters. Based on the R2 value obtained from both the models we select RandomForestRegressor model for fitting. The model is then saved using pickle to deploy it on heroku. 

![image](https://user-images.githubusercontent.com/63156822/133221564-39c8fb23-09bc-4240-8916-ac865184e009.png)

After website is deployed using heroku the user can access it from the link. The user will input the various parameters required for calculating the price and based on the inputs the model will predict the price and display it.

![image](https://user-images.githubusercontent.com/63156822/133221441-dfdefdd6-c3f2-43fb-b631-3a93ad47bd9a.png)


## About
The project has been designed as part of the evaluation scheme of my college course UCS757 - Building Innovative Systems. This project involves the use of Machine Learning to predict the price of an AirBnb based on the user inputs.The dataset used was https://www.kaggle.com/stevezhenghp/airbnb-price-prediction. The data is preprocessed, visualised and then fitted on two models. The RandomForestRegressor is pickled to predict the price since it gave better results. In thisw way the project has been created using the various concepts taught to us in our course curriculum.

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── templates
│   ├── index.html
│   ├── main.html	
├── README.md
├── app.py
├── AirBnB_Predictor.ipynb		
├── file.pkl
├── Procfile
├── requirements.txt
├── Novelty.pdf
├── Flowchart_DS.pdf
├── I_O_Screenshots_DS.pdf
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 

## Future Scope

* Use multiple Algorithms
* Optimize Flask app.py
* Imporve Front-End 
