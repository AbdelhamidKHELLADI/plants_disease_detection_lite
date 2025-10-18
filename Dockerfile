FROM python:3.11-slim

WORKDIR /app

COPY ./models/mobilenet_int8.tflite /app/models/mobilenet_int8.tflite
COPY ./streamlit_app.py /app/streamlit_app.py
COPY ./class_names.txt /app/class_names.txt

RUN pip install --no-cache-dir streamlit ai-edge-litert pillow numpy

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
