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

def transfer(model,image):
    INPUT_HEIGHT = 512   
    INPUT_WIDTH = 256
    scale = min(
        INPUT_HEIGHT / image.shape[0],
        INPUT_WIDTH / image.shape[1])
    
    image = cv2.resize(image, None,fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    off_h = INPUT_HEIGHT - image.shape[0]
    off_w = INPUT_WIDTH - image.shape[1]

    model_input = torch.tensor(image.astype('float32').transpose(2, 0, 1))
    model_input = F.pad(model_input, (off_w, 0, off_h, 0))

    model_output = model(model_input.unsqueeze(0))
    model_output = model_output[0]/255.0
    model_output = torch.max(model_output, torch.tensor(0.0))
    model_output = torch.min(model_output, torch.tensor(1.0))
    model_output = model_output.cpu().numpy().transpose(1, 2, 0)
    model_output = model_output[off_h:, off_w:, :]
    return model_output

def display(image):
    output= transfer(model,image)
    col1, col2 = st.columns(2)
    col1.header("Original")
    col1.image(image, use_column_width=True)
    col2.header("Output")
    col2.image(output, use_column_width=True)

if __name__=='__main__':
    with st.container():
        st.markdown("<h1 style='text-align: center; color: red;'>Art-and-The-Artist🎨</h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: white;'></h6>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        
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

    if option== "(1) VanGogh":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\vangogh.pth'))
        model.eval()
        display(image)
    elif option== "(2) Picasso":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\picasso.pth'))
        model.eval()
        display(image)
    elif option== "(3) Dali":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\picasso.pth'))
        model.eval()
        display(image)
    elif option== "(4) Rembradnt":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\rembradnt.pth'))
        model.eval()
        display(image)
    else:
        pass

    


