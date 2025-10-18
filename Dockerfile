FROM python:3.11-slim AS builder

WORKDIR /app

COPY ./models/mobilenet_int8.tflite /app/models/mobilenet_int8.tflite
COPY ./src/app.py /app/app.py
COPY ./class_names.txt /app/class_names.txt

RUN pip install --no-cache-dir \
    streamlit \
    ai-edge-litert \
    pillow \
    numpy

FROM gcr.io/distroless/python3-debian12

WORKDIR /app
COPY --from=builder /app /app

COPY --from=builder /usr/local/lib/python3.11/site-packages \
              /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/streamlit /usr/local/bin/streamlit
COPY --from=builder /usr/local/bin/python3 /usr/local/bin/python3

EXPOSE 7860
ENV PORT=7860

ENTRYPOINT ["/usr/local/bin/streamlit"]
CMD ["run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
