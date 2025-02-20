from PIL import Image
import torch
from torch.utils import data 
import numpy as np
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import streamlit as st
from efficientnet_pytorch import EfficientNet

####ensemble
class CombinedModel(nn.Module):  
    ''' EDLM '''
    def __init__(self, num_classes):  
        super(CombinedModel, self).__init__()  
        
        # ResNet Backbone  
        self.resnet = torchvision.models.resnet18(pretrained=True)  
        num_ftrs_resnet = self.resnet.fc.in_features  
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes) 
        
        # EfficientNet Backbone  
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs_efficientnet = self.efficientnet._fc.in_features  
        self.efficientnet._fc = nn.Linear(num_ftrs_efficientnet, num_classes)  
        
        # Densenet Backbone
        self.densenet = torchvision.models.densenet121(pretrained=True)  # 
        num_ftrs_densenet = self.densenet.classifier.in_features  
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):  
        result_resnet = self.softmax(self.resnet(x))
        result_efficientnet = self.softmax(self.efficientnet(x))
        result_densenet = self.softmax(self.densenet(x))
        
        output = (result_resnet + result_densenet + result_efficientnet ) / 3
        return output  

num_classes = 2  # 
model = CombinedModel(num_classes) 

# 1. 
def load_model():
    model.load_state_dict(torch.load('./EDLM.pth'))   # 
    model.eval()  # 
    return model

model = load_model()
        
# 2. 
st.title("Classification of Renal Fibrosis with Shear Wave Elastography Image")  # 
st.markdown("""
<style>
    /*  */
    .big-font {
        font-size:24px !important;
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True) 
st.write("Welcome to intelligent renal fibrosis assement.")  # 

# 3. 
uploaded_file = st.file_uploader("Choose one SWE image(jpg)...", type="jpg")  # 
if uploaded_file is not None:
    # 4. 
    image = Image.open(uploaded_file)  # 
    st.image(image, caption='Uploaded Image.', width=300, use_column_width=False)  # 
    st.write("Classifying...")
    # 5.
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 
        transforms.ToTensor(),  #
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # 
    ])

    image = preprocess(image)  # 
    image = image.unsqueeze(0)  #

    # 6. 
    with torch.no_grad():  # 
        output = model(image)  # 
        _, predicted_class = torch.max(output, 1)  # 

    # 7.   
    class_mapping = {0: "mild", 1: "moderate-severe"}

    # 8. 
    predicted_label = class_mapping[predicted_class.item()]
    with st.container():
        st.markdown(f'<p class="big-font">The predicted classification is: {predicted_label}</p>', 
                    unsafe_allow_html=True)  # 
