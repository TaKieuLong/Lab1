import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------------------------
# Bước 1: Kiểm tra môi trường
# ---------------------------
print("TensorFlow Version:", tf.__version__)

# ---------------------------
# Bước 2: Tải mô hình MobileNet V2
# ---------------------------
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)
print("✅ Model loaded successfully!")

# ---------------------------
# Bước 3: Tải hình ảnh và tiền xử lý
# ---------------------------
image_url ="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Felis_silvestris_silvestris_Luc_Viatour.jpg/310px-Felis_silvestris_silvestris_Luc_Viatour.jpg"
save_path = "L:/PY/pug.jpg"  # Đổi sang thư mục hợp lệ


# Lấy tên ảnh từ đường dẫn
image_name = os.path.basename(save_path)

# Kiểm tra thư mục, nếu chưa có thì tạo
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Tải ảnh về máy
response = requests.get(image_url)
if response.status_code == 200:
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"✅ Đã tải ảnh về: {image_name}")
else:
    print("❌ Lỗi tải ảnh!")

# Mở ảnh và resize về 224x224
image = Image.open(save_path).resize((224, 224))

# Hiển thị ảnh
plt.imshow(image)
plt.axis('off')
plt.title(f"Ảnh: {image_name}")
plt.show()

# 2. Tiền xử lý hình ảnh
def preprocess_image(image):
    """
    Chuyển đổi ảnh về numpy array, chuẩn hóa về khoảng [0,1]
    và ép kiểu float32 để tương thích với TensorFlow.
    """
    image = np.array(image, dtype=np.float32) / 255.0  # Ép kiểu float32
    return image[np.newaxis, ...]    # Thêm batch dimension

processed_image = preprocess_image(image)
print("✅ Image preprocessed successfully!")

# ---------------------------
# Bước 4: Dự đoán ảnh
# ---------------------------
# 1. Thực hiện dự đoán
predictions = model(processed_image).numpy()[0]
predicted_class = np.argmax(predictions)  # Lấy lớp có xác suất cao nhất

print("🔎 Predicted class index:", predicted_class)

# 2. Tải danh sách nhãn từ ImageNet
labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", labels_url)

with open(labels_path, "r") as f:
    labels = f.read().splitlines()

# ⚠️ Danh sách ImageNet bắt đầu từ index 1, nhưng mô hình dự đoán index từ 0
corrected_index = predicted_class + 1
predicted_label = labels[corrected_index] if corrected_index < len(labels) else "Unknown"

print(f"✅ Ảnh: {image_name} | Dự đoán: **{predicted_label}**")

# ---------------------------
# Bước 5: Hiển thị kết quả trên ảnh
# ---------------------------
plt.imshow(image)
plt.title(f"Ảnh: {image_name}\nDự đoán: {predicted_label}")
plt.axis('off')
plt.show()
