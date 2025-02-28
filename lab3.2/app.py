from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load biến môi trường
load_dotenv()

# Lấy KEY từ .env
API_KEY = os.getenv('KEY')

# Hàm load mô hình
def load_classification_model():
    model_name = "facebook/deit-base-distilled-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    return feature_extractor, model

# Hàm phân loại ảnh
def classify_image(image_path, feature_extractor, model):
    image = Image.open(image_path).convert("RGB")
    # Resize ảnh về đúng kích thước 224x224
    image = image.resize((224, 224))

    # Truyền thêm tham số size=224 vào feature_extractor
    inputs = feature_extractor(images=image, return_tensors="pt", size=224)
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

# Khởi tạo Flask
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load mô hình
feature_extractor, model = load_classification_model()

# API nhận ảnh và phân loại
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        predicted_label = classify_image(filepath, feature_extractor, model)
        return jsonify({"label": predicted_label, "api_key": API_KEY})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)