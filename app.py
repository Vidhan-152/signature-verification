import streamlit as st
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# ---------------- MODEL DEFINITIONS (same as training) ----------------
class SignatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = SignatureCNN()

    def forward(self, img1, img2):
        return self.feature_extractor(img1), self.feature_extractor(img2)

# ---------------- LOAD MODEL ----------------
device = "cpu"

checkpoint = torch.load("signature_verifier.pth", map_location="cpu")

model = SiameseNetwork().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

THRESHOLD = checkpoint["threshold"]

# ---------------- IMAGE LOADER ----------------
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (160, 100))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    return img

# ---------------- STREAMLIT UI ----------------
st.title("Signature Verification System")

st.write("Upload two signature images to verify whether they are **Genuine** or **Forged**.")

col1, col2 = st.columns(2)

with col1:
    img1_file = st.file_uploader("Upload Reference Signature", type=["png", "jpg", "jpeg"])

with col2:
    img2_file = st.file_uploader("Upload Test Signature", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = preprocess_image(img1_file)
    img2 = preprocess_image(img2_file)

    if st.button("Verify Signature"):
        with torch.no_grad():
            f1, f2 = model(img1, img2)
            dist = F.pairwise_distance(f1, f2).item()

        st.write(f"### Distance: `{dist:.4f}`")
        st.write(f"### Threshold: `{THRESHOLD:.4f}`")

        if dist <= THRESHOLD:
            st.success("Signature is GENUINE")
        else:
            st.error("Signature is FORGED")
