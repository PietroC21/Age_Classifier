import numpy as np
import os
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from keras.models import Sequential
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import imghdr



dir_path = 'data'
image_extensions = ['jpg', 'png', 'jpeg', 'bpm']
for image_class in os.listdir(dir_path):
    for image in os.listdir(os.path.join(dir_path, image_class)):
        image_path = os.path.join(dir_path, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_extensions:
                print(f'Image {image} is not a valid image. Deleting...')
                os.remove(image_path)
        except Exception as e:
            print(f'Image {image} is not a valid image. Deleting...')
            os.remove(image_path)


#Create a dataset from the images
data = tf.keras.utils.image_dataset_from_directory('data')
#Make it numoy iterable
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#Plot the images
#Woman = 1 , Men = 0
print(batch[0].shape)
fig, ax = plt.subplots(ncols=4, figsize=(5, 5))
for idx, img in enumerate(batch[0][:4] ):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

""" ###################### Pre Processing ######################"""
#Scale values to a range of 0 to 1 before feeding them to the CNN model.
#Normalize pixel values between 0 and 1
batch = data_iterator.next()
data = data.map(lambda x, y: (x/255.0, y))
scaled_batch = data.as_numpy_iterator().next()
print(scaled_batch[0].max(), scaled_batch[0].min())



train_size  = int(0.6 * len(data))
val_size  = int(0.2 * len(data))+1
test_size  = int(0.2 * len(data))

train = data.take(train_size)
val = data.skip(test_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


""" ###################### CNN Model ######################"""
#Create the model
model = Sequential([
    Conv2D(16, (3,3),1,  activation='relu', input_shape=(256, 256 ,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Train the model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

#Save the model
model.save('CNN.h5')


#Evaluate the model
pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

for batch in  test.as_numpy_iterator():
    x , y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)
print('Precision: ', pre.result().numpy(), 'Recall: ', rec.result().numpy(), 'Accuracy: ', acc.result().numpy())
