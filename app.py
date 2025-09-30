import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

device = torch.device("cpu")
print(f"Using device: {device}")

data_dir = 'data/train'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory {data_dir} not found. Check your file structure.")
class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)
print(f"Loaded {num_classes} classes: {class_names[:5]}...")

class_label_map = {
    'Apple___healthy': 'Healthy Apple',
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Apple Cedar Rust',
    'Tomato___healthy': 'Healthy Tomato',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Two-Spotted Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
}

model = models.resnet18(weights=None).to(device)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model_path = 'resnet18.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Available models: {os.listdir('.')} (look for .pth files)")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    upload_dir = 'static/uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    image_path = os.path.join(upload_dir, image_file.filename)
    image_file.save(image_path)
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            raw_prediction = class_names[predicted.item()]
            confidence_score = confidence.item()

            friendly_prediction = class_label_map.get(raw_prediction, raw_prediction.replace('_', ' ').title())
            message = f"The {friendly_prediction.lower()} condition is detected with {confidence_score:.1%} confidence."
        
        os.remove(image_path)
        
        return jsonify({'message': message, 'confidence': f"{confidence_score:.4f}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)