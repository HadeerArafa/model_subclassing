import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPool2D,AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model



def Block(inputs ,filters ,stage, kernel_size=3, pool_size=2 , strides=2,repetitions = 2):
    """
    # vgg consist of some repeated blocks each block can has two or three Conv2D layer followed by a MaxPool2D layer
    
    # Arguments
        inputs : can be an input from a previous layer or tf.keras.layers.Input
        filters : the number of filter to use in conv layers 
        kernel_size : default  3
        pool_size : default  2 
        strides : default  2
        repetitions : each block in vgg can consist of two ot three repetitions of Conv2D layer
        stage : a variabel to name our layers

    # Returns
        Output tensor for the block.
    
    """
    x = inputs

    for i in range(repetitions):
        
        x = Conv2D( filters = filters 
                                   ,kernel_size = kernel_size
                                   ,activation='relu'  
                                   ,padding='same'
                                   ,name = f'conv{stage}_{i}')(x)

    x = MaxPool2D(pool_size = pool_size
                   ,strides = strides
                   , name = f'maxpool{stage}')(x)
    return x




def VGG16(include_top = True ,weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          num_classes=1000):
    """
    # Arguments
    include_top: whether to include the fully-connected
        layer at the top of the network.
    weights: one of `None` (random initialization)
        or "imagenet" (pre-training on ImageNet).
    input_tensor: to use as image input for the model.
    input_shape: shape of the input
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the last convolutional layer.
    classes:  number of classes to classify images
        into

    """
    if weights not in {'imagenet' , None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    
    if input_tensor == None and input_shape == None :
        raise ValueError('should enter atleast the shape of the input')
    
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor)
            

    
    # Creating blocks of VGG with the following 
    # (filters, kernel_size, repetitions) configurations
    block_a = Block(inputs = img_input ,filters = 64,  kernel_size = 3, repetitions = 2 ,stage = 1)
    block_b = Block(inputs = block_a ,filters = 128, kernel_size = 3, repetitions = 2 ,stage = 2)
    block_c = Block(inputs = block_b ,filters = 256, kernel_size = 3, repetitions = 3 ,stage = 3)
    block_d = Block(inputs = block_c ,filters = 512, kernel_size = 3, repetitions = 3 ,stage = 4)
    block_e = Block(inputs = block_d ,filters = 512, kernel_size = 3, repetitions = 3 ,stage = 5)
    
    x = block_e
    # Classification head
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(block_e)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(block_e)
    
    model = Model(img_input, x, name='vgg16')
    
    if weights == 'imagenet':
        if include_top:
            weights_path = 'pretrained_models_weights/vgg16/vgg16_weights_top.h5'
        else:
            weights_path = 'pretrained_models_weights/vgg16/vgg16_weights_notop.h5'
        
        model.load_weights(weights_path)
    return model