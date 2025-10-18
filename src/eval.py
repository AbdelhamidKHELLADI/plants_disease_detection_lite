import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory

train_dir="/media/data/plants_diseases_dataset/train"
val_dir="/media/data/plants_diseases_dataset/valid"
img_size=(224,224)
batch_size=32

train_ds=image_dataset_from_directory(train_dir,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      label_mode="categorical")

val_ds=image_dataset_from_directory(val_dir,
                                      image_size=img_size,
                                      batch_size=batch_size,
                                      label_mode="categorical")
def evaluate_model(model, dataset, class_names):
    """
    Evaluate a trained Keras model on a dataset.

    Args:
        model: Trained Keras model
        dataset: tf.data.Dataset (e.g. val_ds)
        class_names: list of class names
    """

    # Get predictions and true labels
    y_true = []
    y_pred = []

    for batch in dataset:
        images, labels = batch
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Overall accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n Accuracy: {acc*100:.2f}%")

from keras.models import load_model
from keras.applications import mobilenet_v3


model = load_model("models/mobileNet_1.keras")
evaluate_model(model, val_ds, class_names=train_ds.class_names)
