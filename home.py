import streamlit as st
import numpy as np
from PIL import Image
from itertools import cycle
import joblib
import json
import cv2
from wavelet import waveletfn

classnum = {}
classname = {}
__model=[]
with open(r'C:\Users\prakhar\Desktop\DsProj3\artifacts\labeldictionary.json', "r") as f:
    classnum = json.load(f)
    classname = {v:k for k,v in classnum.items()}
with open(r'C:\Users\prakhar\Desktop\DsProj3\artifacts\saved_model.pkl', 'rb') as f:
    __model = joblib.load(f)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://media.istockphoto.com/photos/background-black-total-grunge-abstract-cement-concrete-paper-texture-picture-id1251205256?b=1&k=20&m=1251205256&s=170667a&w=0&h=o4N-O5lZywE2lBuauTZX6NiutPCNbKAnuKaTSKWM0Lo=")
    }
    </style>
    """,
    unsafe_allow_html=True
)
def header(url):
     st.markdown(f'<p style="color:#E0B589;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
html_temp = '''<div style="background-color: #1f5e6e;padding:10px"><h2 style="color:white;text-align:center;">Celebrity Recogniser ML App </h2></div>'''
st.markdown(html_temp,unsafe_allow_html=True)
header("Built with Python and Streamlit by Prakhar")
st.subheader("The site will take an input image and recognise whether any of the following celebrities is present in it or not. This project has been made as part of the evaluative component of UCS757 - Building Innovative Systems")
filteredImages=["https://static.toiimg.com/thumb/msid-81665718,width-1200,height-900,resizemode-4/.jpg",
                "data:image/jpeg;base64,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",
                "https://i.guim.co.uk/img/media/9b089d0d5d0939056d2be35001310adc0f355895/428_273_3890_2334/master/3890.jpg?width=1200&height=900&quality=85&auto=format&fit=crop&s=1fd30fc391fa761da462dff91bc39977",
                "https://static.toiimg.com/thumb/msid-54865792,width-1200,height-900,resizemode-4/.jpg",
                "https://img.etimg.com/thumb/msid-55109838,width-1200,height-900,imgsize-145876,overlay-etpanache/photo.jpg",
                "https://m.economictimes.com/thumb/msid-69951568,width-1200,height-900,resizemode-4,imgsize-132920/salmank.jpg"]
caption=["Aamir Khan","Deepika Padukone","Emma Watson","Kevin Hart","Leonardo Di Caprio","Salman Khan"]
idx = 0 
cols = cycle(st.columns(3))
for idx, filteredImage in enumerate(filteredImages):
    next(cols).image(filteredImage, width=150, caption=caption[idx])
input = st.file_uploader("Upload Image",type="jpeg")

if input is not None:
    imgp=Image.open(input)
    st.image(imgp,caption="Image Uploaded")

    if st.button('PREDICT'):
        face_cascade = cv2.CascadeClassifier(r'C:\Users\prakhar\Desktop\DsProj3\haar-cascade-files-master\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(r'C:\Users\prakhar\Desktop\DsProj3\haar-cascade-files-master\haarcascade_eye.xml')
        img = cv2.cvtColor(np.array(imgp), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces):
           
            cropped_faces = []
            for (x,y,w,h) in faces:
                    grayimage = gray[y:y+h, x:x+w]
                    coloredimage = img[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(grayimage)
                    if len(eyes) >= 2:
                        cropped_faces.append(coloredimage)
            if len(cropped_faces):
                imgs = cropped_faces
                result = []
                for img in imgs:
                    scalled_raw_img = cv2.resize(img, (32, 32))
                    img_har = waveletfn(img, 'db1', 5)
                    scalled_img_har = cv2.resize(img_har, (32, 32))
                    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
                    len_image_array = 32*32*3 + 32*32
                    final = combined_img.reshape(1,len_image_array).astype(float)
            
                if ((classname[__model.predict(final)[0]]) =='salman_khan'):
                    st.title("The input image is recognised as Salman Khan")
                elif ((classname[__model.predict(final)[0]]) =='aamir_khan'):
                    st.title("The input image is recognised as Aamir Khan")
                elif ((classname[__model.predict(final)[0]]) =='kevin_hart'):
                    st.title("The input image is recognised as Kevin Hart")
                elif ((classname[__model.predict(final)[0]]) =='leonardo_di_caprio'):
                    st.title("The input image is recognised as Leonardo Di Caprio")
                elif ((classname[__model.predict(final)[0]]) =='deepika_padukone'):
                    st.title("The input image is recognised as Deepika Padukone")
                elif ((classname[__model.predict(final)[0]]) =='emma_watson'):
                    st.title("The input image is recognised as Emma Watson")
                else:
                    st.title("The input image cannot be recognised as any of the above celebrities. Kindly check the face is either clearly visible or is of one of the above mentioned celebrities")
                st.header("Legend:")
                st.subheader(classnum)
                st.header("Table:")
                st.write(np.around(__model.predict_proba(final)*100,2))
        else:
            st.title("The input image cannot be recognised as any of the above celebrities. Kindly enter a clearly visible image")
      


