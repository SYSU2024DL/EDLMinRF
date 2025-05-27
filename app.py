from PIL import Image
import torch
from torch.utils import data 
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import streamlit as st
from efficientnet_pytorch import EfficientNet

####ensemble
class CombinedModel(nn.Module):  
    def __init__(self, num_classes):  
        super(CombinedModel, self).__init__()  
             
        # ResNet Backbone  
        self.resnet = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  
        num_ftrs_resnet = self.resnet.fc.in_features  
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes) 
        
        # EfficientNet Backbone  
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs_efficientnet = self.efficientnet._fc.in_features  
        self.efficientnet._fc = nn.Linear(num_ftrs_efficientnet, num_classes)  
        
        # Densenet Backbone
        self.densenet = torchvision.models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)    
        num_ftrs_densenet = self.densenet.classifier.in_features  
        self.densenet.classifier = nn.Linear(num_ftrs_densenet, num_classes) 
        
    def forward(self, x):  
        result_resnet = self.resnet(x)
        result_efficientnet = self.efficientnet(x)
        result_densenet = self.densenet(x)      
        output = (result_resnet + result_densenet + result_efficientnet ) / 3
        return output     

num_classes = 2  # 
model = CombinedModel(num_classes) 

# 1. 
def load_model():
    model.load_state_dict(torch.load('./EDLM.pth'))   
    model.eval()  
    return model

model = load_model()
        
# 2. 
st.title("Classification of Renal Fibrosis with Shear Wave Elastography Image")  
st.markdown("""
<style>
    /* ####### */
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
        probabilities = F.softmax(output, dim=1)

    # 7.    
    class_mapping = {0: "mild", 1: "moderate-severe"}

    # 8. 
    predicted_label = class_mapping[predicted_class.item()]
    with st.container():
        st.markdown(f'<p class="big-font">The predicted outcome is: {predicted_label}</p>', 
                    unsafe_allow_html=True)  # 
        
        # 9. 
        st.write("### Classification probabilities Visualization")  
        
        # 
        labels = list(class_mapping.values())
        sizes = [probabilities[0][i].item() for i in range(len(class_mapping))]
        explode = [0.1 if i == predicted_class.item() else 0 for i in range(len(class_mapping))]  # 
        
        # 
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # 
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            autopct='%1.2f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999']  # 
        )
        
        # 
        plt.setp(autotexts, size=12)
        plt.setp(texts, size=12)
        
        
        # 
        ax.axis('equal')  
        
        # 
        st.pyplot(fig)
