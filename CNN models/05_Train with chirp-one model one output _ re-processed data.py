import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

dir = 'C:/Users/bono2/OneDrive - Illinois Institute of Technology/Mac/04_sinusoidal + chirp _repreprocessed/'
# dir = '/Volumes/KleeFD/dataset/window_normalized/Window Size/one thirds/'

# Load Dataset
train_X = np.load(dir + 'train_X.npy', mmap_mode = 'r')
# train_Y = (np.load(dir + 'train_Y.npy')[:,0:2]).reshape(-1,2)
train_Y = (np.load(dir + 'train_Y.npy')).reshape(-1,5)

print(train_Y[0:10])

# Make boa - freq tied label
Y = np.zeros((len(train_Y), 1), dtype = '<U10')
for i in range(len(train_Y)):
    Y[i] = (str(train_Y[i,0])+', '+ str(train_Y[i, 1]))
train_Y = Y
print(train_Y[0:10])


val_X = np.load(dir + 'val_X.npy')
# val_Y = (np.load(dir + 'val_Y.npy')[:,0:2]).reshape(-1,2)
val_Y = (np.load(dir + 'val_Y.npy')).reshape(-1,5)

Y = np.zeros((len(val_Y), 1), dtype = '<U10')
for i in range(len(val_Y)):
    Y[i] = (str(val_Y[i,0])+', '+ str(val_Y[i, 1]))
val_Y = Y

print(train_Y[0:10])
print(val_Y[0:10])
# exit()

# To categorical -> one hot encoding
OHE = OneHotEncoder(sparse = False)
train_Y = OHE.fit_transform(train_Y)
val_Y = OHE.transform(val_Y)

print(OHE.categories_)

# from_generator
def generator():
    for i in range(train_Y.shape[0]):
        yield (train_X[i], train_Y[i])

dataset = tf.data.Dataset.from_generator(generator, (tf.float64,tf.float32), ((200,200,1), (16)))
dataset = dataset.batch(8)

# Model (From MIT-BIH)
input1 = tf.keras.layers.Input(shape = (200,200,1), name = 'input1')

hidden1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(input1)
BN1 = tf.keras.layers.BatchNormalization()(hidden1)
MP1 = tf.keras.layers.MaxPooling2D()(BN1)

hidden2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(MP1)
BN2 = tf.keras.layers.BatchNormalization()(hidden2)
MP2 = tf.keras.layers.MaxPooling2D()(BN2)

hidden3 = tf.keras.layers.Conv2D(64, (5,5), activation='relu', padding = 'same')(MP2)
BN3 = tf.keras.layers.BatchNormalization()(hidden3)
MP3 = tf.keras.layers.MaxPooling2D()(BN3)

hidden4 = tf.keras.layers.Conv2D(64, (5,5), activation='relu', padding = 'same')(MP3)
BN4 = tf.keras.layers.BatchNormalization()(hidden4)
MP4 = tf.keras.layers.MaxPooling2D()(BN4)

hidden5 = tf.keras.layers.Conv2D(128, (5,5), activation='relu', padding = 'same')(MP4)
BN5 = tf.keras.layers.BatchNormalization()(hidden5)
MP5 = tf.keras.layers.MaxPooling2D()(BN5)

hidden6 = tf.keras.layers.Conv2D(128, (5,5), activation='relu', padding = 'same')(MP5)
BN6 = tf.keras.layers.BatchNormalization()(hidden6)

hidden7 = tf.keras.layers.Conv2D(128, (5,5), activation='relu', padding = 'same')(BN6)
BN7 = tf.keras.layers.BatchNormalization()(hidden7)

flatten1 = tf.keras.layers.Flatten()(BN7)

Dense1 = tf.keras.layers.Dense(120, activation = 'relu')(flatten1)

output = tf.keras.layers.Dense(16, activation = 'softmax', name = 'output')(Dense1)

model = tf.keras.Model(inputs = input1, outputs = output)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Callbacks
model_save_dir = "C:/Users/bono2/Desktop/Pitching Airfoil/models/Chirp"

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True, verbose = 1) 

weights = {0:2.5, 1:2.5, 2:2.5, 3:2.5, 4:2.5, 5:2.5, 6:2.5, 7:2.5, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1}

for i in range(10):
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_dir + "/iter"+str(i)+".keras", save_best_only = True, verbose = 1)
    model.fit(dataset.repeat(), epochs = 10000, steps_per_epoch = 1130, class_weight = weights\
                    ,validation_data = (val_X, val_Y), callbacks=[early_stopping_cb, checkpoint_cb])