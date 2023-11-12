import numpy as np
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import os

ratios = os.listdir(r'C:\Users\bono2\Desktop\Pitching Airfoil\dataset\window_normalized\Window Size')

first_iter = True
result = {}

for ratio in ratios:

    dataset_dir = 'C:/Users/bono2\Desktop/Pitching Airfoil/dataset/window_normalized/Window Size/' + str(ratio)

    test_X = np.load(dataset_dir + r'\test_X.npy')
    test_Y = (np.load(dataset_dir + r'\test_Y.npy')).reshape(-1,2)
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

    model_dir = 'C:/Users/bono2/Desktop/Pitching Airfoil/models/Testing effect of window size/'+ratio
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
    result[ratio] = np.mean(accuracy)
    print("="*20)
    print(ratio, ":",accuracy)
    print("AVERAGED :", np.mean(accuracy))

names = ['one thirds', 'one fourth', 'one fifth', 'one sixth', 'one seventh', 'one eighth',\
          'one nineth', 'one tenth', 'one eleventh', 'one twelveth', 'one thirteenth', 'one fourteenth', 'one fifteenth']

for name in names:
    print(name,':',result[name])


print(result)
import pickle

with open(r'C:\Users\bono2\Desktop\Pitching Airfoil\models\Testing effect of window size\result.pickle', 'wb') as file:
    pickle.dump(result, file)

