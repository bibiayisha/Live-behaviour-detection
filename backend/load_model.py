# import os
# from sklearn.tree import DecisionTreeClassifier
import numpy as np
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import to_categorical, plot_model

CLASSES_LIST = [
    "Argue",
    "Eating In Class",
    "Explainig The Subject",
    "Hand Raise",
    "Holding Book",
    "Holding Mobile",
    "Reading Book",
    "Sitting On Desk",
    "Writing On Board",
    "Writing On Textbook"
]
def create_LRCN_model():
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'), input_shape = (16, 128, 128, 3)))

    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    # model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(32))

    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    ########################################################################################################################

    # Display the models summary.
    # model.summary()
    
    # Return the constructed LRCN model.
    return model

def load_model():
    print('[load_model] calling create_LRCN_model')
    model = create_LRCN_model()
    # model = tensorflow.keras.models.load_model('C:\\Users\\Ayisha BiBi\\Downloads\\behaviour-tracking\\model\\LRCN_model___Date_Time_2022_11_06__21_22_49___Loss_0.6157196164131165___Accuracy_0.7961783409118652.h5')
    print('[load_model] loading h5 weights')
    model = tensorflow.keras.models.load_model('model\LRCN_model___Date_Time_2022_11_06__21_22_49___Loss_0.6157196164131165___Accuracy_0.7961783409118652.h5')
    # plot_model(model, to_file = 'LRCN_model_structure_plot.png', show_shapes = True, show_layer_names = True)
    # print(model)
    return model

if __name__ == '__main__':
    # model in this file
    model = create_LRCN_model()
    model.summary()
    X = np.random.rand(1, 16, 128, 128, 3)
    y = model.predict(X)
    print(X.shape, y.shape)
    print(y)

    # model from h5 file
    model = tensorflow.keras.models.load_model('..\\model\LRCN_model___Date_Time_2022_11_06__21_22_49___Loss_0.6157196164131165___Accuracy_0.7961783409118652.h5')
    model.summary()
    X = np.random.rand(1, 16, 128, 128, 3)
    y = model.predict(X)
    print(X.shape, y.shape)
    print(y)
