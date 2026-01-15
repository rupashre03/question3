import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="Real-time Image Classification",
    layout="centered",
)

st.title("ResNet-18")

# Load ImageNet labels
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    return requests.get(url).text.splitlines()

# Load pretrained model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

labels = load_labels()
model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Webcam input
st.subheader("Capture Image")
img = st.camera_input("Take a photo")

if img is not None:
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Captured Image")


    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output[0], dim=0)

    # Top 5 predictions
    top5_prob, top5_idx = torch.topk(probs, 5)

    st.subheader("Top 5 Predictions")

    df = pd.DataFrame({
        "Label": [labels[i] for i in top5_idx],
        "Probability": [float(p) for p in top5_prob],
    })

    st.dataframe(df, use_container_width=True)

else:
    st.info("Click **Take a photo** to start classification.")
