from keras import losses
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical

def get_model(n_filters):
    model=Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(filters=n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=2*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(filters=2*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=4*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(filters=4*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(filters=4*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=8*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(filters=8*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=8*n_filters,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=397,activation="softmax"))
    opt=Adam(lr=0.001)
    model.compile(loss=losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    return model
