import tensorflow as tf
from tensorflow.keras.layers import Input ,Conv2D , BatchNormalization , MaxPooling2D ,AveragePooling2D,Activation , DepthwiseConv2D , ZeroPadding2D,Flatten,Dense
from tensorflow.keras.models import Model


def conv_bn(inputs,
           filters,
           kernel_size,
           strides=1,
           stage=None,
           padding='valid'
           ):
    """
    # block for conv+batchnormalization
    # Arguments:
        - inputs : an input tensor from the previous layer or from tf.keras.layers.Inputs
        - filters : number of filters
        - kernal_size : size of the filters
        - stage : used for naming
        - padding : default valid
        - stride : default 1
    """
    
    x = Conv2D( filters , 
                kernel_size , 
                padding= padding , 
                strides=strides ,
                name = f'conv{stage}')(inputs)
    x = BatchNormalization(name = f'bn_{stage}')(x)
    x = Activation('relu', name = f'relu_{stage}')(x)
    
    return x


def conv_dw(inputs,
           pointwise_filters,
           kernel_size,
           strides=1,
           stage=None,
           padding='valid'
           ):
    """
    # block for conv+batchnormalization
    # Arguments:
        - inputs : an input tensor from the previous layer or from tf.keras.layers.Inputs
        - filters : number of filters
        - kernal_size : size of the filters
        - stage : used for naming
        - padding : default valid
        - stride : default 1
    """

    x = DepthwiseConv2D(kernel_size = kernel_size , 
                        strides = strides ,
                        
                        padding= padding , name = f'dw_{stage}')(inputs)
    
    x = BatchNormalization(name = f'bn0_{stage}')(x)
    x = Activation('relu', name = f'relu0_{stage}')(x)
    
    x = Conv2D( filters = pointwise_filters , 
                kernel_size = 1 ,
                strides=1 ,
                padding= 'valid' , 
                name = f'conv1-1_{stage}')(x)
    x = BatchNormalization(name = f'bn1_{stage}')(x)
    x = Activation('relu', name = f'relu1_{stage}')(x)
    
    return x




def MobileNet(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: 
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
        dropout: dropout rate
        classes: optional number of classes to classify images into, 

    Returns:
        A Keras model instance.

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
        
    x = ZeroPadding2D((1, 1),)(img_input) # before adding these the output dimension was 111 so i had to add a zeropadding with padding = 1
    
    x = conv_bn(inputs = x , filters = 32 , kernel_size = 3 , strides = 2 ,stage = 1)
    
    x = conv_dw(inputs = x , pointwise_filters = 64 , kernel_size = 3 , padding = 'same'  , stage = 2)

    x = conv_dw(inputs = x , pointwise_filters = 128 , kernel_size = 3 , padding = 'same' , strides = 2 , stage = 3)

    x = conv_dw(inputs = x , pointwise_filters = 128 , kernel_size = 3 , padding = 'same' , stage = 4)

    x = conv_dw(inputs = x , pointwise_filters = 256 , kernel_size = 3 , padding = 'same' , strides = 2, stage = 5)

    x = conv_dw(inputs = x , pointwise_filters = 256 , kernel_size = 3 , padding = 'same' , stage = 6)

    x = conv_dw(inputs = x , pointwise_filters = 512 , kernel_size = 3 , padding = 'same' , strides = 2 , stage = 7)

    
    for i in range(5):
        x = conv_dw(inputs = x , pointwise_filters = 512 , kernel_size = 3 , padding = 'same' , stage = f'8_{i}')
    
    x = conv_dw(inputs = x , pointwise_filters = 1024 , kernel_size = 3 , padding = 'same' , strides = 2  , stage = 9)

    x = conv_dw(inputs = x , pointwise_filters = 1024 , kernel_size = 3 , padding = 'same' , strides = 1  , stage = 10)

    if include_top == True:
        x = AveragePooling2D(pool_size = 2 , strides = 1)(x)
        x = Flatten()(x)
        x = Dense(units = classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = AveragePooling2D()(x)
        elif pooling == 'max':
            x = MaxPooling2D()(x)
    
    model = Model(inputs = img_input , outputs = x)
    
    if weights == 'imagenet':
        if include_top:
                weights_path = 'pretrained_models_weights/mobilenet/mobilenet_weights_top.h5'
        else:
            weights_path = 'pretrained_models_weights/mobilenet/mobilenet_weights_notop.h5'

        model.load_weights(weights_path , by_name=True, skip_mismatch=True)
    return model    