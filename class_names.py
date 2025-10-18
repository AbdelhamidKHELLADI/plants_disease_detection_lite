import tensorflow as tf

from keras.utils import image_dataset_from_directory

train_dir="/media/data/plants_diseases_dataset/train"

img_size=(224,224)
batch_size=32

train_ds=image_dataset_from_directory(train_dir,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    label_mode="categorical")

class_names=train_ds.class_names
print(class_names)
with open("class_names.txt","w") as f:
    for c in class_names:
        f.write(c+"\n")
