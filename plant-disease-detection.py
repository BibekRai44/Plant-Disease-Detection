import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms


model_path = 'plant-disease-model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

print(checkpoint.keys())

model_key = model_path

model = checkpoint[model_key]
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


classes = ['class1', 'class2', 'class3', ...]

def predict_disease(image):
   
    image = transform(image).unsqueeze(0)

  
    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = classes[predicted_idx.item()]

    return predicted_label

def main():
    st.title("Leaf Disease Prediction")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predicted_disease = predict_disease(image)
        st.write("Predicted Disease:", predicted_disease)

if __name__ == "__main__":
    main()
