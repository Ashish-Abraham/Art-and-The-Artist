import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2


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

def display(model,image):
    output= transfer(model,image)
    col1, col2 = st.columns(2)
    col1.header("Original")
    col1.image(image, use_column_width=True)
    col2.header("Output")
    col2.image(output, use_column_width=True)