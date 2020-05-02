import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plot

def load_img(path):
    image = Image.open(path)
    image = image.resize((28, 28))
    image = np.dot(np.asarray(image), np.array([1 / 3, 1 / 3, 1 / 3]))
    image /= 255
    image = 1 - image
    return image.reshape((1, 28 * 28))

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plot.plot(epochs, loss, 'm', label='Training loss')
plot.plot(epochs, val_loss, 'b', label='Validation loss')
plot.title('Training and validation loss')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.legend()
plot.show()
plot.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plot.plot(epochs, acc, 'm', label='Training accuracy')
plot.plot(epochs, val_acc, 'b', label='Validation accuracy')
plot.title('Training and validation accuracy')
plot.xlabel('Epochs')
plot.ylabel('Accuracy')
plot.legend()
plot.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)