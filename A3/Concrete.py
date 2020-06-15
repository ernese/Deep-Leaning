#Import Required Libraries

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
#import cv2
import tensorflow as tf
import keras
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions



# Create a Window.
MyWindow = Tk() # Create a window
MyWindow.title("Concrete crack classification")
MyWindow.geometry('650x300') # Set the size of the Windows
 
# Create the Custom Methods for Processing the Images/Video using DL model

def openImg(filename):    
    global imageLabel
    global my_image2
    my_image2 = ImageTk.PhotoImage(Image.open(filename))    
    imageLabel.grid_forget()
    imageLabel.configure(image=my_image2)
    imageLabel.image = my_image2
    imageLabel.grid(row = 0, column = 2, columnspan = 2, rowspan = 2, padx = 5, pady = 5)
    
    
def BttnOpen_Clicked():    
    global imageLabel
    global fileImage
    messagebox.showinfo("Info", "Open Button Clicked")
    #Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Images files","*.jpg"),("VideoFiles","*.mp4"),("all files","*.*")))
    fileImage = file
    messagebox.showinfo("File Selected", file)
    #Open image function
    openImg(file)
    return fileImage

def BttnProcess_structure():
    global L4
    global fileImage
    global result
    global x
    
    #print (fileImage)
    messagebox.showinfo("Info", "Process Button Clicked")
    #Load pretrain model
    modelStructure = load_model('C:/Users/ernes/Documents/uni/2020Aut/Deep_Learning/Ass3/A3_weights_final/Multiclass_weights.10-0.11.hdf5')

    #Downloading the image
    img = image.load_img(fileImage, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    #Classification

    preds = modelStructure.predict(x)
    classifcationResult = preds
    
    z = np.argmax(classifcationResult)
    if z == 0:
            result = "Bridge Deck"
    elif z == 1:
            result = "Pavement"
    elif z == 2:
            result = "Wall"

    #Updating the label L4
    L4.configure(text = result)  # Update the Label text on the Window
    return result
    return x

    
def BttnProcess_cracked():
    global L6
    global x
    global result
    global preds
    
    #print (fileImage)
    messagebox.showinfo("Info", "Process Button Clicked")

    #Classification

    if result == "Bridge Deck":
        modelcracked = load_model('C:/Users/ernes/Documents/uni/2020Aut/Deep_Learning/Ass3/A3_weights_final/Deck_weights.18-0.20.hdf5')
        preds = modelcracked.predict_classes(x)
        z = int(preds)
        if z==0:
            #Updating the label L6
            result2 = "Cracked"
            L6.configure(text = result2)     
        else:
            result2 = "unCracked"
            L6.configure(text = result2)
            
    elif result == "Pavement":
        modelcracked = load_model('C:/Users/ernes/Documents/uni/2020Aut/Deep_Learning/Ass3/A3_weights_final/Pavement_weights.19-0.18.hdf5')
        preds = modelcracked.predict_classes(x)
        z = int(preds)
        if z==0:
            #Updating the label L6
            result2 = "Cracked"
            L6.configure(text = result2)     
        else:
            result2 = "unCracked"
            L6.configure(text = result2)
            
    elif result == "Wall":
        modelcracked = load_model('C:/Users/ernes/Documents/uni/2020Aut/Deep_Learning/Ass3/A3_weights_final/Wall_weights.10-0.10.hdf5')
        preds = modelcracked.predict_classes(x)
        z = int(preds)
        if z==0:
            #Updating the label L6
            result2 = "Cracked"
            L6.configure(text = result2)     
        else:
            result2 = "unCracked"
            L6.configure(text = result2)
            
    return preds
    return x
    return result

def refresh():
    global x
    global result
    global destroy
    global imageLabel
    global preds
    global L4
    global L6

    imageLabel.grid_forget()

    imageLabel.configure(text = "Image")

    L4.configure(text= "")
    L6.configure(text = "")

    try:
        x.destroy()
        result.destroy()
        fileImage.destroy()
        preds.destroy()
    except:
        pass
    
 
#GUI components
    
#Classification of structure Label
L3 = Button(text="Identify structure", command=BttnProcess_structure, padx = 37, pady = 10)
L3.grid(row = 0, column = 0, sticky = W, pady = 2)
 
L4 = Label(text="Structure is", font= "Helvetica 9 bold italic")
L4.grid(row = 0, column = 1, pady = 2 , sticky = W)

#Classification of cracked or uncracked Label
L5 = Button(text="Identify cracked/Uncracked", command=BttnProcess_cracked, padx = 10, pady = 10)
L5.grid(row = 1, column = 0, sticky = W, pady = 2)

L6 = Label(text="cracked/uncracked", font= "Helvetica 9 bold italic")
L6.grid(row = 1, column = 1, pady = 2, sticky = W) 

#Invisible Label
Lx = Label(MyWindow, text = " ")
Lx.grid(row = 2, column = 0, sticky = W, columnspan = 2)
 
#ImageLabel Label
imageLabel = Label(text = "Image" )
imageLabel.grid(row = 0, column = 2, columnspan = 2, rowspan = 2, padx = 115, pady = 125)

#Upload image Label
L1 = Label(MyWindow, text = "Click to Open an Image", font=("Arial Bold", 9))
L1.grid(row = 2, column = 2, sticky = E)

L2 = Button(MyWindow,text="Open Image", command = BttnOpen_Clicked, font=("Arial Bold", 9), fg="blue")
L2.grid(row = 2, column = 3, sticky = W)

#Refresh
L7 = Button(MyWindow,text="Refresh", command = refresh, font=("Arial Bold", 9), fg="Black")
L7.grid(row = 2, column = 4, sticky = W, padx = 3)




# Calling the maninloop()
MyWindow.mainloop()
