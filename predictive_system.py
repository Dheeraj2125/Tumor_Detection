import numpy as np
import pickle
from matplotlib.pyplot import imshow
from PIL import Image

#loading saved model
loaded_model=pickle.load(open(r'C:\Users\Anirudh\OneDrive\Desktop\Summer Project\trained_model.sav','rb'))

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


img = Image.open(r"C:\Users\Anirudh\OneDrive\Desktop\Summer Project\yes\Y2.jpg")
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = loaded_model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))