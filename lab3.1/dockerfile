# Sử dụng Python base image
FROM python:3.8-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . /app

# Cập nhật pip lên phiên bản mới nhất
RUN pip install --upgrade pip

# Cài đặt các thư viện cần thiết
RUN pip install tensorflow numpy pillow flask

# Cài đặt các phụ thuộc hệ thống nếu cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libmagic-dev

# Mở cổng 5000 cho ứng dụng Flask
EXPOSE 5000

# Định nghĩa biến môi trường
ENV NAME=World

# Sao chép tệp saveModel.py vào container
COPY saveModel.py /app/

# Chạy saveModel.py để tải mô hình
RUN python saveModel.py

# Chạy app.py sau khi đã tải mô hình
CMD ["python", "app.py"]