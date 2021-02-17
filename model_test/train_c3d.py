# -*- coding:utf-8 -*-
import keras
from model_test.models import c3d_model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from model_test.schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import os
import random
import matplotlib

# TODO categorical -> binominal??
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def process_batch(lines, img_path, train=True):
    IMG_WIDTH = 171
    IMG_HEIGHT = 128

    num = len(lines)
    batch = np.zeros((num, 16, IMG_HEIGHT, IMG_WIDTH, 3), dtype='float32')
    labels = np.zeros(num, dtype='int')

    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol) - 1
        imgs = os.listdir(img_path + path)
        imgs.sort(key=str.lower)
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            # is_flip = random.randint(0, 1)
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # openCV stores data color as BGR
                # TODO image resize
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # if is_flip == 1:
                #     image = cv2.flip(image, 1)
                # 16 frame을 다 넣었다는 증거!!
                batch[i][j][:][:][:] = image
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                image = cv2.imread(img_path + path + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                batch[i][j][:][:][:] = image
            labels[i] = label
    return batch, labels


# TODO 어떻게 RGB 전처리를 한거죠?
def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.

    # inputs[..., 0] /= 255.0
    # inputs[..., 1] /= 255.0
    # inputs[..., 2] /= 255.0

    return inputs


def generator_train_batch(train_txt, batch_size, num_classes, img_path):
    ff = open(train_txt, 'r')  # train_txt = train_list.txt
    lines = ff.readlines()
    num = len(lines)  # num = 2847
    while True:
        new_line = []
        index = [n for n in range(num)]  # 0 ~ 2846  # 첫번째에는 [0], 두번째 [1] ... [2846]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])  # new_line = lines list를 섞음
        for i in range(int(num / batch_size)):  # 177
            a = i * batch_size  # 0, 16, 32, 48
            b = (i + 1) * batch_size  # 16, 32, 48, 64
            x_train, x_labels = process_batch(new_line[a:b], img_path, train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)  # One-hot Encoding
            x = np.transpose(x, (0, 2, 3, 1, 4))
            yield x, y


def generator_val_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(new_line[a:b], img_path, train=False)
            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    img_path = '/home/pirl/PycharmProjects/cnnTest/FrameImg/'
    train_file = 'newTrainlist.txt'
    test_file = 'newTestlist.txt'
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)  # Confusing name : why val?

    # hyper parameter
    num_classes = 2
    batch_size = 16
    epochs = 14
    lr = 0.01

    model = c3d_model(True)  # train mode: True, test mode: False
    model.summary()

    # gpu set
    modelFromGpu = multi_gpu_model(model, gpus=2)

    # optimizer Adam
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)

    modelFromGpu.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = modelFromGpu.fit_generator(generator_train_batch(train_file, batch_size, num_classes, img_path),
                                         steps_per_epoch=train_samples // batch_size,
                                         epochs=epochs,
                                         callbacks=[onetenth_4_8_12(lr)],
                                         validation_data=generator_val_batch(test_file,
                                                                             batch_size, num_classes, img_path),
                                         validation_steps=val_samples // batch_size,
                                         verbose=1)
    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history, 'results/')
    save_history(history, 'results/')

    model.save_weights('results/weights_c3d_lr001.h5')


if __name__ == '__main__':
    main()
