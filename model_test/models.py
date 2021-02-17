'''

작정자: 김희섬(gmltja@naver.com)
모델 개요:
    1. 8개의 conv.layer 와 5개의 pooling layer로 구성됨.
    2. batch normalization을 사용하므로, 학습 모드와 테스트 모드로 구분됨.(isTrain=False이면  테스트 모드)

'''




import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation, GlobalAveragePooling3D, ZeroPadding3D
from keras.regularizers import l2
from keras.models import Model
from keras import backend



def c3d_model(isTrain):

    input_shape = (128,171,16,3)
    # weight_decay = 0.005
    nb_classes = 2

    inputs = Input(input_shape)


    # First
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same')(inputs)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)
    x = MaxPool3D(strides=2,pool_size=(2,2,1),padding='same')(x)

    # 2nd
    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same')(x)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)
    x = MaxPool3D(strides=2, pool_size=(2, 2, 2),padding='same')(x)

    # 3rd
    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same')(x)
    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)
    x = MaxPool3D(strides=2, pool_size=(2, 2, 2),padding='same')(x)

    # 4th
    x = Conv3D(512,(3,3,3),strides=(1,1,1),padding='same')(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)
    x = MaxPool3D(strides=2, pool_size=(2, 2, 2),padding='same')(x)

    # 5th
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)
    # x = MaxPool3D(strides=2, pool_size=(2, 2, 2), padding='same')(x)

    # add zero padding
    x = ZeroPadding3D(padding=1)(x)
    # additional conv. layer
    x = Conv3D(1024, (3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    # batchLayer = BatchNormalization()
    # batchLayer.trainable = isTrain
    x = BatchNormalization(trainable=isTrain)(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation = 'softmax')(x)

    model = Model(inputs, x)
    # model.summary()



    return model

# c3d_model(False)


# get_3rd_layer_output = K.function([model.layers[0].input,
#                                    backend.learning_phase()],
#                                   [model.layers[3].output])
