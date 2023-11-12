import numpy as np
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import os

result = {}

base_window_ratio = os.listdir(r'C:\Users\bono2\Desktop\Pitching Airfoil\dataset\window_normalized\Sparse Sensing')

first_iter = True

for ratio in base_window_ratio:

    dir = 'C:/Users/bono2/Desktop/Pitching Airfoil/dataset/window_normalized/Sparse Sensing/' + str(ratio)
    strides = os.listdir(dir)
    stride_result = {}
    for stride in strides:
        dataset_dir = 'C:/Users/bono2/Desktop/Pitching Airfoil/dataset/window_normalized/Sparse Sensing/' + str(ratio)
        dataset_dir += '/' + str(stride)

        # Load Data
        test_X = np.load(dataset_dir + '/test_X.npy')
        test_Y = (np.load(dataset_dir + '/test_Y.npy')).reshape(-1,2)
        Y = np.zeros((len(test_Y), 1), dtype = '<U10')
        for i in range(len(test_Y)):
            Y[i] = (str(test_Y[i,0])+', '+ str(test_Y[i, 1]))
        test_Y = Y

        OHE = OneHotEncoder(sparse = False)
        if first_iter:
            test_Y = OHE.fit_transform(test_Y)
        else:
            test_Y = OHE.transform(test_Y)
        
        actual = np.argmax(test_Y, axis = 1)
        prediction = []

        model_dir = 'C:/Users/bono2/Desktop/Pitching Airfoil/models/Testing effect of sparse sensing/' + ratio +'/'+ stride
        model = []

        for i in range(1,11):
            model.append(keras.models.load_model(model_dir + '/iter '+str(i)+'.h5'))

        for i in range(10):
            # Get Predictions (reverse to_categorical)
            pred_proba = model[i].predict(test_X)
            prediction.append(np.argmax(pred_proba, axis = 1))

        accuracy = []

        for i in range(10):
            accuracy.append(Counter(prediction[i] == actual)[True] / len(prediction[i]))

        
        stride_result[stride] = np.mean(accuracy)

    result[ratio] = stride_result
        
names = ['one_thirds_window', 'one_fourth_window', 'one_fifth_window', 'one_sixth_window', 'one_seventh_window']


for name in names:
    temp = result[name]
    print('-'*20)
    print("Base :",name)
    temp_keys = list(temp.keys())
    temp_vals = list(temp.values())
    for key, val in zip(temp_keys, temp_vals):
        print(key,'-',val)


print(result)
import pickle

with open(r'C:\Users\bono2\Desktop\Pitching Airfoil\models\Testing effect of sparse sensing\result.pickle', 'wb') as file:
    pickle.dump(result, file)