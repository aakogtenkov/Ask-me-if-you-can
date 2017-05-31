
# coding: utf-8

# # Лабораторная №3

# Вначале импортируем нужные вещи:

# In[1]:

from __future__ import print_function
import numpy
numpy.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.utils import np_utils

from scipy.misc import imresize


# Ниже приведенный код обрабатывает тестовые картинки (приводя их размер к 64х64) и разбивает их на пакеты по 5000 картинок.

# In[6]:

X_train = numpy.load('test.npy')
print(X_train.shape)
BUFF_SIZE = 5000
IMAGE_SIZE = (64, 64)
for i in range(0, len(X_train), BUFF_SIZE):
    tmp = numpy.zeros((min(BUFF_SIZE, len(X_train)-i), 64, 64))
    for j in range(i, min(len(X_train), i + BUFF_SIZE)):
        #print(i, j, X_train[j])
        tmp[j - i] = imresize(X_train[j], IMAGE_SIZE)
    tmp = numpy.dstack(tmp)
    tmp = numpy.rollaxis(tmp, 2)
    print(tmp.shape)
    #tmp = numpy.vstack((tmp, y_train[i:min(len(y_train), i+BUFF_SIZE)]))
    print(i // BUFF_SIZE)
    numpy.save(str(i // BUFF_SIZE) + '-normalized_64_test.npy', tmp)
del X_train


# В ячейках ниже я считываю тренировочный сет, преобразовываю к картинки к размеру 64х64 и разбиваю на пакеты. Для более удобного чтения (и записи, в принципе, тоже), ответы хранятся отдельно.

# In[6]:

data = numpy.load('train.npy')
X_train = data[:, 0]
y_train = data[:, 1]


# In[7]:

print(len(X_train[1]), len(X_train))


# In[8]:

BUFF_SIZE = 5000
IMAGE_SIZE = (64, 64)
for i in range(0, len(X_train), BUFF_SIZE):
    tmp = numpy.zeros((min(BUFF_SIZE, len(X_train)-i), 64, 64))
    for j in range(i, min(len(X_train), i + BUFF_SIZE)):
        #print(i, j, X_train[j])
        tmp[j - i] = imresize(X_train[j], IMAGE_SIZE)
    tmp = numpy.dstack(tmp)
    tmp = numpy.rollaxis(tmp, 2)
    print(tmp.shape, y_train.shape)
    #tmp = numpy.vstack((tmp, y_train[i:min(len(y_train), i+BUFF_SIZE)]))
    print(i // BUFF_SIZE)
    numpy.save(str(i // BUFF_SIZE) + '-normalized_64_pictures.npy', tmp)
    numpy.save(str(i // BUFF_SIZE) + '-normalized_64_answers.npy', y_train[i:min(len(y_train), i+BUFF_SIZE)])


# Следующая ячейка является подготовительной: в ней можно удалить все загруженные ранее данные из памяти, также в ней происходит создание словаря: неудобному номеру иероглифа сопоставляется число от 0 до 499.

# In[2]:

#del X_train
#del data
#del y_train
data = numpy.load('train.npy')
Y_train = data[:, 1]
class_values = dict()
for elem in Y_train:
    if not(elem in class_values):
        class_values[elem] = len(class_values)
del Y_train
del data


# Одна из нейронок, которые я использовал (всего их было не меньше 5):

# In[3]:

model = Sequential()
model.add(Convolution2D(24, 3, 3,
                        border_mode='valid',
                        input_shape=(64, 64, 1)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3,
                        border_mode='same'))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(len(class_values)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer=Adadelta(),
            metrics=['categorical_accuracy'])


# Ячейка для обучения:

# In[4]:

for ep in range(8):
    for batch in range(0, 29):
        X_train = numpy.load(str(batch) + '-normalized_64_pictures.npy')
        Y_train = numpy.load(str(batch) + '-normalized_64_answers.npy')

        for i in range(len(Y_train)):
            Y_train[i] = class_values[Y_train[i]] 
        Y_train = np_utils.to_categorical(Y_train, len(class_values))

        #print(X_train.shape, Y_train.shape)
        X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
        X_train = X_train.astype('float32')
        X_train /= 255
    #X_train.shape = (X_train.shape[0], 32 * 32)
    #print(X_train.shape, len(X_train[0]), len(X_train[1]), X_train[0].shape)
    #for i in range(len(X_train)):
    #    X_train[i] = X_train[i].reshape((70, 70, 1))
    #print(X_train.shape, Y_train.shape)
    #X_train = numpy.zeros((10000, 70, 70, 1))
    #X_train.shape = (X_train.shape[0], 70, 70, 1)

        history = model.fit(X_train, Y_train,
                        batch_size=150, nb_epoch=1,
                        verbose=0, validation_data=(X_train, Y_train))
        score = model.evaluate(X_train, Y_train, verbose=0)
        print(ep + 1, batch, score)
        del X_train
        del Y_train
    print("LET'S TEST IT!")
    for batch in range(29, 34):
        X_train = numpy.load(str(batch) + '-normalized_64_pictures.npy')
        Y_train = numpy.load(str(batch) + '-normalized_64_answers.npy')

        for i in range(len(Y_train)):
            Y_train[i] = class_values[Y_train[i]] 
        Y_train = np_utils.to_categorical(Y_train, len(class_values))

        X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
        X_train = X_train.astype('float32')
        X_train /= 255

        #history = model.fit(X_train, Y_train,
        #                batch_size=128, nb_epoch=10,
        #                verbose=0, validation_data=(X_train, Y_train))
        score = model.evaluate(X_train, Y_train, verbose=0)
        print(batch, score)
        del X_train
        del Y_train


# Еще одна нейронка:

# In[3]:

model = Sequential()
model.add(Convolution2D(28, 3, 3,
                        border_mode='valid',
                        input_shape=(64, 64, 1)))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(len(class_values)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer=Adadelta(),
            metrics=['categorical_accuracy'])


# Ячейка для дообучения:

# In[16]:

for ep in range(1):
    for batch in range(33, -1, -1):
        X_train = numpy.load(str(batch) + '-normalized_64_pictures.npy')
        Y_train = numpy.load(str(batch) + '-normalized_64_answers.npy')

        for i in range(len(Y_train)):
            Y_train[i] = class_values[Y_train[i]] 
        Y_train = np_utils.to_categorical(Y_train, len(class_values))

        #print(X_train.shape, Y_train.shape)
        X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
        X_train = X_train.astype('float32')
        X_train /= 255
        history = model.fit(X_train, Y_train,
                        batch_size=150, nb_epoch=1,
                        verbose=0, validation_data=(X_train, Y_train))
        #score = model.evaluate(X_train, Y_train, verbose=0)
        print(ep + 1, batch)
        del X_train
        del Y_train
    print("LET'S TEST IT!")
    for batch in range(0, 34, 3):
        X_train = numpy.load(str(batch) + '-normalized_64_pictures.npy')
        Y_train = numpy.load(str(batch) + '-normalized_64_answers.npy')

        for i in range(len(Y_train)):
            Y_train[i] = class_values[Y_train[i]] 
        Y_train = np_utils.to_categorical(Y_train, len(class_values))

        X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
        X_train = X_train.astype('float32')
        X_train /= 255

        #history = model.fit(X_train, Y_train,
        #                batch_size=128, nb_epoch=10,
        #                verbose=0, validation_data=(X_train, Y_train))
        score = model.evaluate(X_train, Y_train, verbose=1)
        print(batch, score)
        del X_train
        del Y_train


# Ячейка для записи предсказаний нейронки в файл:

# In[17]:

output = open('pred_64_13.txt', 'w')
output.write("Id,Category\n")
c = 1
for batch in range(0, 9):
    X_test = numpy.load(str(batch) + '-normalized_64_test.npy')
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)
    X_test = X_test.astype('float32')
    X_test /= 255
    pred = model.predict_classes(X_test, batch_size = 100, verbose=1)
    for elem in pred:
        good = False
        output.write(str(c) + ',')
        for _c in class_values:
            if class_values[_c] == elem:
                if good:
                    print("ERROR")
                output.write(str(_c) + '\n')
                good = True
        c += 1
    del X_test
    del pred
output.close()


# Еще ячейка для дообучения:

# In[20]:

for batch in range(1, 34):
    X_train = numpy.load(str(batch) + '-normalized_64_pictures.npy')
    Y_train = numpy.load(str(batch) + '-normalized_64_answers.npy')

    for i in range(len(Y_train)):
        Y_train[i] = class_values[Y_train[i]] 
    Y_train = np_utils.to_categorical(Y_train, len(class_values))

    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    history = model.fit(X_train, Y_train,
                        batch_size=128, nb_epoch=1,
                        verbose=0, validation_data=(X_train, Y_train))
    score = model.evaluate(X_train, Y_train, verbose=0)
    print(batch, score)
    del X_train
    del Y_train


# Ячейка для сравнения ответов нескольких нейронок и выбор лучшего ответа:

# In[6]:

file1 = open('pred_64_1.txt', 'r')
file2 = open('pred_64_4.txt', 'r')
file3 = open('pred_64_common_8.txt', 'r')
file_ans = open('pred_64_common_11.txt', 'w')
s1 = file1.readline()
s2 = file2.readline()
s3 = file3.readline()
file_ans.write(s1)
s1 = file1.readline()
s2 = file2.readline()
s3 = file3.readline()
c1 = 0
c2 = 0
while len(s1) > 0:
    s1 = s1.split(',')
    s2 = s2.split(',')
    s3 = s3.split(',')
    file_ans.write(s1[0] + ',')
    ans1 = int(s1[1])
    ans2 = int(s2[1])
    ans3 = int(s3[1])
    if not(ans1 == ans2 == ans3):
        c1 += 1
    if ans1 == ans2:
        file_ans.write(str(ans1))
    elif ans1 == ans3:
        file_ans.write(str(ans1))
    elif ans2 == ans3:
        file_ans.write(str(ans2))
    else:
        file_ans.write(str(ans3))
        print(s1[0], ans1, ans2, ans3)
        c2 += 1
    file_ans.write('\n')
    s1 = file1.readline()
    s2 = file2.readline()
    s3 = file3.readline()
print(c1, c2)
file_ans.close()
file1.close()
file2.close()
file3.close()

