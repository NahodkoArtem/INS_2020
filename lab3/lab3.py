import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(test_targets)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 6
num_val_samples = len(train_data) // k
num_epochs = 40
all_scores = []
mae_histories = []
loss_array = []
val_loss_array = []
mae_array = []
val_mae_array = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
    mae_array.append(H.history['mae'])
    val_mae_array.append(H.history['val_mae'])
    loss_array.append(H.history['loss'])
    val_loss_array.append(H.history['val_loss'])

    print(H.history.keys())
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Train loss')
    plt.plot(epochs, val_loss, 'b', label='Valid loss')
    plt.title('Train and Valid loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    mae = H.history['mae']
    val_mae_graph = H.history['val_mae']
    plt.plot(epochs, mae, 'bo', label='Train mae')
    plt.plot(epochs, val_mae_graph, 'b', label='Valid mae')
    plt.title('Train and valid mae')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

epochs = range(1, num_epochs + 1)
plt.plot(epochs, np.mean(loss_array, axis=0), 'bo', label='Training loss')
plt.plot(epochs, np.mean(val_loss_array, axis=0), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, np.mean(mae_array, axis=0), 'bo', label='Training mae')
plt.plot(epochs, np.mean(val_mae_array, axis=0), 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

print(np.mean(all_scores))

