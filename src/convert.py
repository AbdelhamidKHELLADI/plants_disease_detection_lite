import tensorflow as tf
import numpy as np
import time
import os
from keras.utils import image_dataset_from_directory

val_dir="/media/data/plants_diseases_dataset/valid"
path="/media/data/plants_diseases_dataset/"
img_size=(224,224)
batch_size=32
INPUT_SHAPE = (224, 224, 3)

keras_model = tf.keras.models.load_model("models/mobileNet_10.keras")


converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("models/mobilenet_int8.tflite", "wb") as f:
    f.write(tflite_model)


def evaluate_models(keras_model, tflite_model_path, dataset):
    """Compare size, speed, and accuracy of Keras vs TFLite model."""
    keras_size = os.path.getsize("models/mobileNet_10.keras") / 1024**2
    tflite_size = os.path.getsize(tflite_model_path) / 1024**2



    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0
    times = []

    for batch_images, batch_labels in dataset.take(50):  
        batch_images = batch_images.numpy()
        batch_labels = tf.argmax(batch_labels, axis=1).numpy()

        for i in range(len(batch_images)):
            input_data = np.expand_dims(batch_images[i], axis=0).astype(np.float32)
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            times.append(time.time() - start)

            pred = np.argmax(output)
            if pred == batch_labels[i]:
                correct += 1
            total += 1

    tflite_acc = correct / total
    avg_time = np.mean(times) * 1000  
    print(f"\nModel sizes:")
    print(f" - Keras: {keras_size:.2f} MB")
    print(f" - TFLite INT8: {tflite_size:.2f} MB")
    print("---------------------------------------------")
    _, keras_acc = keras_model.evaluate(dataset, verbose=0)
    print(f"\n Keras Model Accuracy: {keras_acc*100:.2f}%")

    print(f" TFLite Model Accuracy: {tflite_acc*100:.2f}%")
    print(f" Avg Inference Time (1 image): {avg_time:.2f} ms")

    print("\n Summary:")
    print(f" - Size reduction: {(1 - tflite_size/keras_size)*100:.1f}%")
    print(f" - Accuracy drop: {(keras_acc - tflite_acc)*100:.2f}%")



val_ds=image_dataset_from_directory(val_dir,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    label_mode="categorical")
evaluate_models(keras_model, "models/mobilenet_int8.tflite", val_ds)
