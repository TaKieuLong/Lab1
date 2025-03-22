from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Tải mô hình MobileNetV2 với trọng số ImageNet
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Đọc danh sách nhãn từ tệp
with open("imagenet_label.txt") as f:
    labels = f.read().splitlines()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy file ảnh từ request
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))  # Kích thước chuẩn cho MobileNetV2
        img_array = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

        # Dự đoán
        prediction = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=3)[0]

        # Chuyển đổi kết quả thành JSON
        result = [{"label": pred[1], "probability": float(pred[2])} for pred in decoded_predictions]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)