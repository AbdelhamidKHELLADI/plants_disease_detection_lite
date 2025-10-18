import tensorflow as tf
import keras
from keras.applications import MobileNetV3Small, mobilenet_v3
from keras import layers,models
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


def preprocess(x):
    return mobilenet_v3.preprocess_input(x)
train_dir="/media/data/plants_diseases_dataset/train"
val_dir="/media/data/plants_diseases_dataset/valid"
path="/media/data/plants_diseases_dataset/"
img_size=(224,224)
batch_size=32
INPUT_SHAPE = (224, 224, 3)
num_epochs=10
print(INPUT_SHAPE)
def get_datasets(path):
    train_dir=path+"/train"
    val_dir=path+"/valid"
    train_ds=image_dataset_from_directory(train_dir,
                                        image_size=img_size,
                                        batch_size=batch_size,
                                        label_mode="categorical")

    val_ds=image_dataset_from_directory(val_dir,
                                        image_size=img_size,
                                        batch_size=batch_size,
                                        label_mode="categorical")
    num_classes=len(train_ds.class_names)
    train_ds=train_ds.map(lambda x, y: (mobilenet_v3.preprocess_input(x), y))
    val_ds=val_ds.map(lambda x, y: (mobilenet_v3.preprocess_input(x), y))

    AUTOTUNE=tf.data.AUTOTUNE
    train_ds=train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds=val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, num_classes




def create_MobileNet(INPUT_SHAPE,NUM_CLASSES):
    base_model=MobileNetV3Small(input_shape=INPUT_SHAPE,
                                include_top=False,
                                weights='imagenet')

    model=models.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES,activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_hist(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():

    train_ds,val_ds,num_classes=get_datasets(path)
    model=create_MobileNet(INPUT_SHAPE,num_classes)
    history=model.fit(train_ds,validation_data=val_ds,epochs=num_epochs)
    model.save(f"models/mobileNet_{num_epochs}.keras")
    plot_hist(history)

if __name__=="__main__":
    main()