import numpy as np
# from tensorflow import keras
from keras import layers, models
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os

ratios = os.listdir(r'C:\Users\bono2\Desktop\Pitching Airfoil\dataset\window_normalized\Window Size')

first_iter = True

for ratio in ratios:
    print("="*30)
    print("WORKING ON RATIO : " + ratio)
    print("="*30)


    dir = 'C:/Users/bono2\Desktop/Pitching Airfoil/dataset/window_normalized/Window Size/' + str(ratio)

    # Load Data
    train_X = np.load(dir + r'\train_X.npy', mmap_mode='r')
    train_Y = (np.load(dir + r'\train_Y.npy')).reshape(-1,2)
    Y = np.zeros((len(train_Y), 1), dtype = '<U10')
    for i in range(len(train_Y)):
        Y[i] = (str(train_Y[i,0])+', '+ str(train_Y[i, 1]))
    train_Y = Y

    val_X = np.load(dir + r'\val_X.npy')
    val_Y = (np.load(dir + r'\val_Y.npy')).reshape(-1,2)
    Y = np.zeros((len(val_Y), 1), dtype = '<U10')
    for i in range(len(val_Y)):
        Y[i] = (str(val_Y[i,0])+', '+ str(val_Y[i, 1]))
    val_Y = Y

    # To categorical -> one hot encoding
    OHE = OneHotEncoder(sparse = False)
    if first_iter:
        train_Y = OHE.fit_transform(train_Y)
    else:
        train_Y = OHE.transform(train_Y)

    val_Y = OHE.transform(val_Y)

    # from_generator
    def generator():
        for i in range(train_Y.shape[0]):
            yield (train_X[i], train_Y[i])

    image_size = train_X.shape[1]

    dataset = tf.data.Dataset.from_generator(generator, (tf.float64,tf.float32), ((image_size,image_size,1), (16)))
    dataset = dataset.batch(8)


    # Model (From MIT-BIH)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_size,image_size,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (5,5), activation='relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (5,5), activation='relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())

    model.add(layers.Dense(120, activation='relu'))

    model.add(layers.Dense(16, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    model_save_dir = 'C:/Users/bono2/Desktop/Pitching Airfoil/models/Testing effect of window size/' + ratio
    os.mkdir(model_save_dir)
    
    for iteration_number in range(10):
        # Callbacks
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True) 
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_dir + "/iter " + str(iteration_number + 1) + ".h5", save_best_only = True)

        weights = {0:2.5, 1:2.5, 2:2.5, 3:2.5, 4:2.5, 5:2.5, 6:2.5, 7:2.5, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1}

        history = model.fit(dataset.repeat(), epochs = 10000, steps_per_epoch = 982, class_weight = weights, validation_data = (val_X, val_Y),\
            callbacks=[early_stopping_cb, checkpoint_cb])