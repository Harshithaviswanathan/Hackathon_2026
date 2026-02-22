import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class_names = ["FAKE", "REAL"]

# Load model
@st.cache_resource
def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, img_tensor):
    img_tensor = transform(img_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
    return probs.cpu().numpy()[0]

def add_noise(img_tensor, epsilon):
    noise = torch.randn_like(img_tensor) * epsilon
    return torch.clamp(img_tensor + noise, 0, 1)

def blur_image(img_tensor, k):
    return TF.gaussian_blur(img_tensor, kernel_size=k)

def fgsm_attack(image, epsilon):
    image = image.unsqueeze(0).clone().detach().to(device)
    image.requires_grad = True
    output = model(image)
    loss = nn.CrossEntropyLoss()(output, torch.tensor([1]).to(device))  # target FAKE
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed, 0, 1).detach()

# UI
st.title("🛡 Deepfake Detection – Robust AI")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Baseline Model", "Robust Model"]
)

attack_choice = st.sidebar.selectbox(
    "Apply Attack",
    ["None", "Gaussian Noise", "Blur", "FGSM Attack"]
)

epsilon = st.sidebar.slider("Noise Strength", 0.0, 0.5, 0.05)
blur_k = st.sidebar.slider("Blur Kernel Size", 3, 15, 5, step=2)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Load correct model
    if model_choice == "Baseline Model":
        model = load_model("best_model.pth")
    else:
        model = load_model("robust_model_multidefense.pth")

    # Convert to tensor (without normalization for attack)
    img_tensor = transforms.ToTensor()(image)

    # Apply attack if selected
    if attack_choice == "Gaussian Noise":
        img_tensor = add_noise(img_tensor, epsilon)
        st.write("Applied Gaussian Noise")

    elif attack_choice == "Blur":
        img_tensor = blur_image(img_tensor, blur_k)
        st.write("Applied Blur")
    
    elif attack_choice == "FGSM Attack":
        img_tensor = fgsm_attack(img_tensor, epsilon)
        st.write("Applied FGSM attack")

    # Convert tensor back to PIL for prediction pipeline
    modified_image = transforms.ToPILImage()(img_tensor)

    # Show modified image
    st.image(modified_image, caption="Image Used for Prediction", use_column_width=True)

    # ✅ Now prediction is done on the correct image
    probs = predict(model, modified_image)

    st.subheader("Prediction")
    st.write(f"REAL Confidence: {probs[1]:.4f}")
    st.write(f"FAKE Confidence: {probs[0]:.4f}")

    predicted_class = class_names[np.argmax(probs)]
    st.success(f"Predicted: {predicted_class}")
