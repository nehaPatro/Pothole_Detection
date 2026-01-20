FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install YOLOv12
RUN git clone https://github.com/sunsmarterjie/yolov12.git
RUN pip install -e yolov12

COPY . .

CMD ["python", "app.py"]
