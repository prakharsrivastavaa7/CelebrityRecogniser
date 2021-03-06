# CelebrityRecogniser
 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [About](#About)
  * [Deployement on Streamlit](#deployement-on-streamlit)
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
  
I used python script - simple-image-download to download images of six celebrities from the net automatically. The six celebrities are Aamir khan, Deepika padukone, Salman Khan, Emma Watson, Kevin Hart and Leonardo Di Caprio. The images are then preprocessed using haar cascade and wavelet transformation function. Finally we train multiple models on then and select the best performing one and pickle it. The ML classifier is created and then webapp is deployed using Streamlit.   
  
![Flowchart](https://user-images.githubusercontent.com/63156822/141940955-7fa82363-f37a-45aa-8b79-b5d0ba95a376.jpeg)


## About
The project has been designed as part of the evaluation scheme of my college course UCS757 - Building Innovative Systems. This project involves the use of Machine Learning to predict the input image recognises which of the celebrities present in the dataset.The dataset used was generated by me through webscrapping the images of six celebrities. The data is preprocessed, visualised and then fitted on multiple models. The Logistic Regression model is pickled to classify the image since it gave better results. In this way the project has been created using the various concepts taught to us in our course curriculum.

## Deployement on Streamlit
The web app is deployed for free through streamlit which supports python app 


[<img target="_blank" src="https://mms.businesswire.com/media/20200616005364/en/798639/23/Streamlit_Logo_%281%29.jpg" width=170>](https://mms.businesswire.com/media/20200616005364/en/798639/23/Streamlit_Logo_%281%29.jpg) 




## Directory Tree 
```
????????? croppedaamir_khan_images
????????? croppeddeepika_padukone_images
????????? croppedemma_watson_images
????????? croppedkevin_hart_images
????????? croppedleonardo_di_caprio_images
????????? croppedsalman_khan_images
????????? haar-cascade-files-master
????????? haarcascade_eye.xml
????????? haarcascade_frontalface_default.xml
????????? labeldictionary.json
????????? README.md
????????? home.py
????????? wavelet.py
????????? setup.sh
????????? Datasetgeneration.ipynb		
????????? ImageClassification.ipynb	
????????? saved_model.pkl
????????? Procfile
????????? requirements.txt
????????? Novelty.pdf
????????? Flowchart.jpeg
????????? Main Page-converted.pdf
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://mms.businesswire.com/media/20200616005364/en/798639/23/Streamlit_Logo_%281%29.jpg" width=170>](https://mms.businesswire.com/media/20200616005364/en/798639/23/Streamlit_Logo_%281%29.jpg)  

[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/1200px-OpenCV_Logo_with_text_svg_version.svg.png" width=170>](https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/1200px-OpenCV_Logo_with_text_svg_version.svg.png)


## Future Scope

* Use multiple Algorithms
* Optimize Streamlit app
* Imporve Front-End 
