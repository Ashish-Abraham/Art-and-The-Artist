import json
from io import BytesIO
from PIL import Image
import os

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2

from transferNet import FinalNet
from utils.functions import transfer, display


if __name__=='__main__':
    # Declare the global variable
    image = None
    with st.container():
        st.markdown("<h1 style='text-align: center; color: red;'>Art-and-The-Artist🎨</h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: white;'></h6>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original =Image.open(uploaded_file)
        
        # picture = st.camera_input("Take a picture")
        # if picture:
        #     st.image(picture)


    # designing sidebar
    image1 = Image.open('../images/combined.png')

    # Display combined image in sidebar
    st.sidebar.image(image1)
    option = st.sidebar.selectbox(
    "Select your style:",
    ("(1) VanGogh", "(2) Picasso", "(3) Dali", "(4) Rembradnt")
    )

    if uploaded_file is None:
        st.write("Please upload an image")
    else:    
        if option== "(1) VanGogh":
            model = FinalNet()
            model.load_state_dict(torch.load('./model_files/vangogh.pth',map_location=torch.device('cpu')))
            model.eval()
            display(model,image,original)
        elif option== "(2) Picasso":
            model = FinalNet()
            model.load_state_dict(torch.load('./model_files/picasso.pth',map_location=torch.device('cpu')))
            model.eval()
            display(model,image,original)
        elif option== "(3) Dali":
            model = FinalNet()
            model.load_state_dict(torch.load('./model_files/picasso.pth',map_location=torch.device('cpu')))
            model.eval()
            display(model,image,original)
        elif option== "(4) Rembradnt":
            model = FinalNet()
            model.load_state_dict(torch.load('./model_files/rembradnt.pth',map_location=torch.device('cpu')))
            model.eval()
            display(model,image,original)
        else:
            pass

    


