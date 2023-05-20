import json
from io import BytesIO
from PIL import Image
import os

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from transferNet import FinalNet

if __name__=='__main__':
    with st.container():
        st.markdown("<h1 style='text-align: center; color: red;'>Art-and-The-Artist🎨</h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center; color: white;'></h6>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image)
            output = model(image)
            output = model_output[0]/255.0
            output = torch.max(output, torch.tensor(0.0))
            output = torch.min(output, torch.tensor(1.0))
            output = output.cpu().numpy().transpose(1, 2, 0)
            output = output[off_h:, off_w:, :]
        # picture = st.camera_input("Take a picture")
        # if picture:
        #     st.image(picture)


    # Load images
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
    elif option== "(2) Picasso":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\picasso.pth'))
        model.eval()
    elif option== "(3) Dali":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\picasso.pth'))
        model.eval()
    elif option== "(4) Rembradnt":
        model = FinalNet()
        model.load_state_dict(torch.load('src\model_files\rembradnt.pth'))
        model.eval()
    else:
        pass
    


