from tensorflow.keras.callbacks import Callback
#from keras.backend import clear_session
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.applications import ResNet50, MobileNet, Xception , DenseNet121, InceptionV3, VGG16, VGG19
#import tensorflow as tf
#from tensorflow.python.framework import ops
#ops.reset_default_graph()
from tensorflow.keras import backend as K

def build_model(mode, model_name = None, model_path = None):

    clear_session()

    if mode == 'train':
        img = Input(shape = (96,96,3)) # ResNet50 minimum size (32,32) for others (128,128)

        if model_name == 'DenseNet121': #Checked and Working

            model = DenseNet121(include_top=False, 
                                weights='imagenet', 
                                input_tensor=img, 
                                input_shape=None, 
                                pooling='avg')

        elif model_name == 'MobileNet': #checked, raised shape error, #Error Resolved, Now working

            model = MobileNet(include_top=True, 
                              weights='imagenet', 
                              input_tensor=img, 
                              input_shape=None, 
                              pooling='avg')

        elif model_name == 'Xception': #Checked and Working

            model = Xception(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='max')

        elif model_name == 'ResNet50': #Image Dimesion size should be high eg 224x224, not sufficient GPU memory resource

            model = ResNet50(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='avg')

        elif model_name == 'InceptionV3': #Checked and Working

            model = InceptionV3(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=(None), 
                             pooling='avg')
        elif model_name == 'VGG19': #to be checked

            model = InceptionV4(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='avg')

        elif model_name == 'VGG16': #Checked and Working
            model = VGG16(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=(None), 
                             pooling='max')

        elif model_name == 'VGG19': #to be checked

            model = VGG19(include_top=False, 
                             weights='imagenet', 
                             input_tensor=img, 
                             input_shape=None, 
                             pooling='avg')


        final_layer = model.layers[-1].output

        dense_layer_1 = Dense(128, activation = 'relu')(final_layer)
        output_layer = Dense(4, activation = 'softmax')(dense_layer_1)

        model = Model(inputs = [img], outputs = output_layer)
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

    elif mode == 'inference':
        model = load_model(model_path)

    return model

