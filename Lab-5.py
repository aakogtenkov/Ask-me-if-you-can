
# coding: utf-8

# # Лабораторная №5: транскрипции

# Сначала приготовим данные для обучения. Для этого в отдельный файл запишем все встречающиеся фонемы.
# Выводится самое длинное слово.

# In[1]:

import numpy


input = open('train.txt', 'r')
output = open('phonems.txt', 'w')
s = input.readline()
d = dict()
max_word = ''
max_tr = ''
while len(s) > 0:
    phonems = (s.split()[1]).split('_')
    elem = s.split()[0]
    if len(elem) > len(max_word):
        max_word = elem
    if len(phonems) > len(max_tr):
        max_tr = phonems
    for ph in phonems:
        if not(ph in d):
            d[ph] = len(d)
            output.write(ph + ' ' + str(len(d) - 1) + '\n')
    s = input.readline()
d['EOF'] = len(d)
output.write('EOF ' + str(len(d) - 1) + '\n')
input.close()
output.close()
print(max_word, len(max_word))
print(max_tr, len(max_tr))


# Теперь создадим numpy-матрицу для всех слов, которую мы будем впоследствии пихать в нейронку. Матрица будет иметь размер (количество слов $ \cdot$ максимальный размер слова $\cdot $ количество букв + EOF).

# In[2]:

NUM_LETTERS = ord('Z') - ord('A') + 1 + 1
MAX_LETTERS = 34
MAX_PHONEMS = 32

input = open('train.txt', 'r')
s = input.readline()
matrix = []
matrix_ans = []
c = 0
while len(s) > 0:
    word = s.split()[0]
    ans = (s.split()[1]).split('_')
    word.replace("'", "")
    if len(word) < MAX_LETTERS and len(ans) < MAX_LETTERS:
        c += 1
        for i in range(0, len(word)):
            cur_symb = [0] * NUM_LETTERS
            cur_symb[ord(word[i]) - ord('A')] = 1
            matrix.append(cur_symb)
        for i in range(len(word), MAX_LETTERS):
            cur_symb = [0] * NUM_LETTERS
            cur_symb[-1] = 1
            matrix.append(cur_symb)
        for i in range(0, len(ans)):
            cur_symb = [0] * len(d)
            cur_symb[d[ans[i]]] = 1
            matrix_ans.append(cur_symb)
        for i in range(len(ans), MAX_PHONEMS):
            cur_symb = [0] * len(d)
            cur_symb[d['EOF']] = 1
            matrix_ans.append(cur_symb)
    s = input.readline()
matrix_ans = numpy.array(matrix_ans)
matrix = numpy.array(matrix)
matrix = numpy.reshape(matrix, (c, MAX_LETTERS, NUM_LETTERS))
matrix_ans = numpy.reshape(matrix_ans, (c, MAX_PHONEMS, len(d)))
input.close()


# In[3]:

numpy.save('word_matrix', matrix)
numpy.save('ans_matrix', matrix_ans)


# In[4]:

print(matrix.shape)
print(matrix_ans.shape)


# Загрузка матриц:

# In[1]:

import numpy

NUM_LETTERS = ord('Z') - ord('A') + 1 + 1
MAX_LETTERS = 34
MAX_PHONEMS = 32

matrix = numpy.load('word_matrix.npy')
matrix_ans = numpy.load('ans_matrix.npy')
input = open('phonems.txt', 'r')
s = input.readline()
d = dict()
while len(s) > 0:
    s = s.split()
    d[s[0]] = int(s[1])
    s = input.readline()
input.close()
del input


# In[5]:

print(matrix.shape)
print(matrix_ans.shape)
print(len(d))
tmp_ans = ''
for j in range(8):
    for k in range(27):
        if matrix[128, j, k] == 1:
            tmp_ans += chr(k + ord('A'))
print(tmp_ans)
tmp_ans = ''
for j in range(7):
    for k in range(40):
        if matrix_ans[128, j, k] == 1:
            for elem in d:
                if d[elem] == k:
                    tmp_ans += elem + '_'
print(tmp_ans)


# Создаем модель:

# In[1]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed
import numpy


# In[3]:

model = Sequential()
model.add(Bidirectional(LSTM(output_dim=128), merge_mode='concat', input_shape=(MAX_LETTERS, NUM_LETTERS)))
model.add(RepeatVector(MAX_PHONEMS))
model.add(LSTM(return_sequences=True, output_dim=128))
model.add(TimeDistributed(Dense(len(d))))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])


# Обучение:

# In[13]:

for ep in range(12):
    model.fit(matrix, matrix_ans, nb_epoch=4, verbose=2)
    model.save('transcriptions_model_l19' + str(ep) + '.h5')


# Другая модель:

# In[11]:

model = Sequential()
model.add(SimpleRNN(128, input_shape=(MAX_LETTERS, NUM_LETTERS)))
model.add(RepeatVector(MAX_PHONEMS))
model.add(Bidirectional(SimpleRNN(128, return_sequences=True)))
model.add(TimeDistributed(Dense(len(d))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['RMSprop'])


# Здесь мы загружаем модель:

# In[11]:

from keras.models import load_model


NUM_LETTERS = ord('Z') - ord('A') + 1 + 1
MAX_LETTERS = 19
MAX_PHONEMS = 19
model = load_model('transcriptions_model_l1913.h5')


# Клетка для тестирования модели:

# In[12]:

while (True):
    word = input().split()[0]
    inp_vect = []
    if word == 'end':
        break
    word.replace("'", "")
    for i in range(0, len(word)):
        cur_symb = [0] * NUM_LETTERS
        cur_symb[ord(word[i]) - ord('A')] = 1
        inp_vect.append(cur_symb)
    for i in range(len(word), MAX_LETTERS):
        cur_symb = [0] * NUM_LETTERS
        inp_vect.append(cur_symb)
    inp_vect = numpy.array(inp_vect)
    inp_vect = numpy.reshape(inp_vect, (1, MAX_LETTERS, NUM_LETTERS))
    ans = model.predict(inp_vect, 1, 0)
    
    for i in range(0, ans.shape[1]):
        max_val = -100
        best_p = -1
        for j in range(0, ans.shape[2]):
            #print(ans[0, i, j])
            if max_val < ans[0, i, j]:
                max_val = ans[0, i, j]
                best_p = j
        for elem in d:
            if d[elem] == best_p:
                print(elem, end = '_')
    print()


# Клетки ниже являются отладочными или не несут особой роли.
# Например, клетку ниже можно запускать для получения информации о количестве слов с определенным количеством фонем/букв.

# In[7]:

input = open('train.txt', 'r')
s = input.readline()
count_letters_in_word = [0] * 34
count_letters_in_ph = [0] * 32
count_need_words = 0
while len(s) > 0:
    word = s.split()[0]
    word.replace("'", "")
    count_letters_in_word[len(word) - 1] += 1
    ans = (s.split()[1]).split('_')
    count_letters_in_ph[len(ans) - 1] += 1
    if len(ans) <= 8 and len(ans) >= 4 and len(word) <= 9 and len(word) >= 5:
        count_need_words += 1
    s = input.readline()
input.close()
print(count_letters_in_word)
print()
print(count_letters_in_ph)
print()
print(count_need_words)


# Это просто копия одной из клеток выше с другими параметрами и некоторыми изменениями. Использовалась для отладки. Оставляю только для истории:

# In[33]:

import numpy

NUM_LETTERS = ord('Z') - ord('A') + 1 + 1
MAX_LETTERS = 19
MAX_PHONEMS = 19

input = open('train.txt', 'r')
s = input.readline()
matrix = []
matrix_ans = []
c = 0
while len(s) > 0:
    word = s.split()[0]
    word.replace("'", "")
    ans = (s.split()[1]).split('_')
    if len(word) <= MAX_LETTERS - 1 and len(ans) <= MAX_PHONEMS - 1:
        c += 1
    for i in range(0, len(word)):
        cur_symb = [0] * NUM_LETTERS
        cur_symb[ord(word[i]) - ord('A')] = 1
        if len(word) <= MAX_LETTERS - 1 and len(ans) <= MAX_PHONEMS - 1:
            matrix.append(cur_symb)
    for i in range(len(word), MAX_LETTERS):
        cur_symb = [0] * NUM_LETTERS
        cur_symb[-1] = 1
        if len(word) <= MAX_LETTERS - 1 and len(ans) <= MAX_PHONEMS - 1:
            matrix.append(cur_symb)
    for i in range(0, len(ans)):
        cur_symb = [0] * len(d)
        cur_symb[d[ans[i]]] = 1
        if len(word) <= MAX_LETTERS - 1 and len(ans) <= MAX_PHONEMS - 1:
            matrix_ans.append(cur_symb)
    for i in range(len(ans), MAX_PHONEMS):
        cur_symb = [0] * len(d)
        cur_symb[d['EOF']] = 1
        if len(word) <= MAX_LETTERS - 1 and len(ans) <= MAX_PHONEMS - 1:
            matrix_ans.append(cur_symb)
    s = input.readline()
matrix_ans = numpy.array(matrix_ans)
matrix = numpy.array(matrix)
matrix = numpy.reshape(matrix, (c, MAX_LETTERS, NUM_LETTERS))
matrix_ans = numpy.reshape(matrix_ans, (c, MAX_PHONEMS, len(d)))
input.close()
print(matrix.shape)
print(matrix_ans.shape)


# In[34]:

numpy.save('word_matrix_l19', matrix)
numpy.save('ans_matrix_l19', matrix_ans)


# Клетка для записи ответа:

# In[18]:

input = open('test.csv', 'r')
output = open('ans.txt', 'w')
s = input.readline()
s = input.readline()
c = 0
percent = 0
prev_percent = 0
while len(s) > 0:
    c += 1
    word = s.split(',')[1]
    inp_vect = []
    word.replace("'", "")
    word = word.rstrip()
    if len(word) >= MAX_LETTERS:
        word = "EMPTY"
    for i in range(0, len(word)):
        cur_symb = [0] * NUM_LETTERS
        #print(word[i], NUM_LETTERS, ord(word[i]) - ord('A'))
        cur_symb[ord(word[i]) - ord('A')] = 1
        inp_vect.append(cur_symb)
    for i in range(len(word), MAX_LETTERS):
        cur_symb = [0] * NUM_LETTERS
        inp_vect.append(cur_symb)
    inp_vect = numpy.array(inp_vect)
    inp_vect = numpy.reshape(inp_vect, (1, MAX_LETTERS, NUM_LETTERS))
    ans = model.predict(inp_vect, 1, 0)
    
    ans_str = []
    for i in range(0, ans.shape[1]):
        max_val = -100
        best_p = -1
        for j in range(0, ans.shape[2]):
            #print(ans[0, i, j])
            if max_val < ans[0, i, j]:
                max_val = ans[0, i, j]
                best_p = j
        if best_p == ans.shape[2] - 1:
            break
        for elem in d:
            if d[elem] == best_p:
                ans_str.append(elem)
    output.write(str(c) + ',' + '_'.join(ans_str) + '\n')
    s = input.readline()
    if percent / 415 > prev_percent:
        print(prev_percent)
        prev_percent += 1
    percent += 1

