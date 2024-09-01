import numpy as np
import pickle
import streamlit as st
from matplotlib.pyplot import imshow
from PIL import Image

#loading saved model
loaded_model=pickle.load(open(r'C:\Users\Anirudh\OneDrive\Desktop\Summer Project\trained_model.sav','rb'))

def names(number):
    if number==0:
        return 'a Tumor'
    else:
        return 'not a tumor'


#create func
def tumor_prediction(input_data) :
    #r"C:\Users\Anirudh\OneDrive\Desktop\Summer Project\yes\Y2.jpg"
    img = Image.open(input_data)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    res = loaded_model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    return (str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))

def main():
    
    
    # giving a title
    st.title('Brain Tumor Identification Web App')
    
    
    # getting the input data from the user
    
    name= st.text_input("Enter Patient Name")
    upload_file=st.file_uploader("Choose A FIle")
    if upload_file is not None:
    # Open the image using PIL
          img = Image.open(upload_file)
# Display the image using Streamlit
          st.image(img, caption='Uploaded MRI Scan Image', use_column_width=True)
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('tumor Test Result'):
        diagnosis = tumor_prediction(upload_file)
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()