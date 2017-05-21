
# coding: utf-8

# # Лабораторная №4

# ## LSA
# 
# Прежде всего считаем файл, полученный в результате парсинга. Запишем в словарь все слова.
# "tttrrr" - это пометка, что началась новая статья.

# In[1]:

import copy

input = open('simplewiki-parsed3.txt', 'r')
word_dict = dict()
now_context = 0
s = input.readline()
c = 0
prev_c = 0
while len(s) > 0:
    s = s.split()
    if len(s) > 0 and s[0] == "tttrrr":
        s = []
        now_context += 1
    for elem in s:
        if not(elem in word_dict):
            wl = len(word_dict)
            word_dict[elem] = wl
    s = input.readline()
    c += 1
    if c // 26000 > prev_c:
        prev_c += 1
        #print(prev_c)
print(now_context, len(word_dict))
input.close()


# Создадим разреженную матрицу. Запишем в нее стартовые значения.

# In[20]:

import numpy
from scipy.sparse import lil_matrix

mas = lil_matrix((len(word_dict), now_context + 2), dtype='float16')
print(mas.shape)
print(mas[2473, 1])


# In[22]:

input = open('simplewiki-parsed3.txt', 'r')
now_context = 0
s = input.readline()
c = 0
prev_c = 0
while len(s) > 0:
    s = s.split()
    if len(s) > 0 and s[0] == "tttrrr":
        s = []
        now_context += 1
    for elem in s:
        mas[word_dict[elem], now_context] += 1
    s = input.readline()
    c += 1
    if c // 26000 > prev_c:
        prev_c += 1
        #print(prev_c)
print(now_context, len(word_dict))
input.close()


# Запустим SVD-разложение.

# In[23]:

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components = 128, n_iter = 8)
mas = svd.fit_transform(mas)


# Сохраним результат. Матрица $mas$ - это матрица, в которой строки соответствуют словам.

# In[24]:

numpy.save('mas-1', mas)
output = open('words-1.txt', 'w')
for elem in word_dict:
    output.write(elem + ' ' + str(word_dict[elem]) + '\n')
output.close()


# Это код для считывания $mas$ и списка слов из файла:

# In[2]:

import numpy
import copy

mas = numpy.load('mas-1.npy')
print(mas.shape)

input = open('words-1.txt', 'r')
word_dict = dict()
s = input.readline()
while len(s) > 0:
    s = s.split()
    num = int(s[1])
    word_dict[s[0]] = num
    s = input.readline()
print(now_context, len(word_dict))
input.close()


# Функции расстояния: обычное евклидова мера, косинусная мера и измененная мера Минковского (мера Минковского подразумевало бы возведение в слишком большую степень, что приводило бы к большой погрешности).

# In[10]:

from math import sqrt

def get_dist(word1, word2, k = 128):
    ind1 = word_dict[word1]
    ind2 = word_dict[word2]
    d = numpy.sum((mas[ind1] - mas[ind2]) * (mas[ind1] - mas[ind2]))
    return d

def get_dist_v(vect, word2, k = 128):
    vect2 = mas[word_dict[word2]]
    d = numpy.sum((vect2 - vect) * (vect2 - vect))
    return d

def get_cos_dist(word1, word2, k = 128):
    ind1 = word_dict[word1]
    ind2 = word_dict[word2]
    dot_prod = numpy.dot(mas[ind1], mas[ind2])
    d1 = sqrt(numpy.dot(mas[ind1], mas[ind1]))
    d2 = sqrt(numpy.dot(mas[ind2], mas[ind2]))
    return (dot_prod / (d1 * d2))

def get_cos_dist_v(vect, word2, k = 128):
    ind2 = word_dict[word2]
    dot_prod = numpy.dot(vect, mas[ind2])
    d1 = sqrt(numpy.dot(vect, vect))
    d2 = sqrt(numpy.dot(mas[ind2], mas[ind2]))
    return (dot_prod / (d1 * d2))

def get_half_minkowski_dist(word1, word2, k = 128, p = 8):
    ind1 = word_dict[word1]
    ind2 = word_dict[word2]
    d = 0
    for i in range(k):
        d += (mas[ind1][i] - mas[ind2][i])**p
    d **= (1/p)
    return d


# В приведенном ниже коде можно тремя разными способами получать значение расстояния, и код будет находить ближайшие 10 слов (в порядке убывания близости).

# In[2]:

def get_nearest(string = "thrones", n = 11, metric = "cos"):
    min_d = [10000000000000000000000000 for i in range(n)]
    ans = ['' for i in range(len(min_d))]
    for word in word_dict:
        if metric == "euclid":
            d = get_dist(string, word)
        elif metric == "cos":
            d = 2 - get_cos_dist(string, word)
        else:
            d = get_half_minkowski_dist(string, word)
        pst = False
        for p in range(0, len(min_d)):
            if not(pst) and d < min_d[p]:
                for j in range(len(min_d) - 1, p, -1):
                    min_d[j] = min_d[j - 1]
                    ans[j] = ans[j - 1]
                min_d[p] = d
                ans[p] = word
                pst = True
    print(string + ": [" + ', '.join(ans) + ']')


# In[5]:

get_nearest('actor')


# Ниже я привожу некоторые слова и выданные программой списки близких слов (согласно заданию, нужно было выбрать 30 слов, но я выбрал около 25, ибо не хватило фантазии. 30 слов приведены для skip-gram модели).
# 
# ### Евклидова метрика:
# 
# putin: [putin, abdel, merkel, dalai, fidel, abkhazia, sarkozy, ossetia, aliyev, goh]
# 
# moscow: [moscow, vladimir, liberation, ukraine, sicily, milan, munich, prussia, prague, constantine]
# 
# lord: [lord, rome, sons, palace, christ, lady, jerusalem, throne, heaven, cross, temple]
# 
# computer: [computer, particular, software, web, standard, happens, systems, warnings, itself, normal, separate]
# 
# siberia: [siberia, leased, russia's, tripoli, stormed, airspace, phan, wooded, xian, surabaya, muhammed]       (казалось бы, при чем тут Триполи??)
# 
# morning: [morning, evening, afternoon, midnight, stayed, arrived, dropped, trip, deep, broke, murder]
# 
# moon: [moon, planet, deep, sky, secret, rise, beauty, sun, journey, dog, 20th]
# 
# nuclear: [nuclear, world's, economic, economy, growth, security, iraq, islamic, afghanistan, industrial, billion]
# 
# ship: [ship, officers, battles, remained, orders, boat, fired, surrender, destroyed, destroy, armed]
# 
# job: [job, stay, bring, deal, complete, rest, lots, accept, thinking, trouble, moment]
# 
# escape: [escape, turns, alive, trapped, falling, stolen, burning, sword, steal, revenge, thrown]
# 
# wikipedia: [wikipedia, please, changes, times, about, palatino, message, been, signpost, your, more]
# 
# outpost: [outpost, dusk, cheering, bombard, receipt, urging, deakin, sloop, steamed, foreground, deed]
# 
# car: [car, market, thames, freight, spa, self, express, cars, f1, races, driver]
# 
# team: [team, won, championship, wmf, foundation, future, win, notifications, across, players, teams]
# 
# 
# hobbit: [hobbit, elves, gollum, dwarves, bilbo, orcs, sauron, hobbits, tolkien, pippin, baggins]
# 
# forest: [forest, wood, garden, counties, yorkshire, gardens, oak, greater, salt, palm, zone]
# 
# dragon: [dragon, sword, monster, lion, quest, spy, hercules, heroes, chase, wears, dreams]
# 
# imperial: [imperial, arrived, fleet, supplies, occupied, losses, commanded, supply, landed, officials, invaded]
# 
# metric: [metric, measurement, martinvl, defacto, centaur, indicated, metrication, measures, templatedata, nuts, attribute]
# 
# empty: [empty, blank, specify, parent, specified, setting, replace, maintenance, defined, window, settings]
# 
# dinosaurs: [dinosaurs, fossils, cretaceous, evolved, fossil, feathers, skull, bones, reptiles, extinction, jurassic]
# 
# actor: [actor, actress, singer, director, player, former, writer, politician, musician, author, composer]        (есть ощущение, что не хватило данных, чтобы программа научилась как следует различать людей творческих профессий)
# 
# light: [light, energy, sun, inside, earth, nature, side, dark, theory, surface, lower]
# 
# jedi: [jedi, sith, unstoppable, puppies, wrath, toon, snowman, smurfs, dante's, bug's, looney]
# 
# september: [september, 20, 21, 12, 26, 23, 30, block, 19, 24, i'm]             (мораль - убирайте даты из текста)
# 
# 
# 
# ### Косинусная метрика:
# 
# putin: [putin, haider, gaza, medvedev, merkel, al-assad, bashar, abdel, jong-un, president, modi]
# 
# moscow: [moscow, ii, basel, yuri, russia, vasili, bratislava, genoa, cologne, lugano, leningrad]
# 
# lord: [lord, knights, king, sons, honourable, son, elder, jerusalem, god, vere, voyage]
# 
# computer: [computer, laptop, processing, minimize, straightforward, computers, reciprocal, fizz, advantages, 4px, used]
# 
# siberia: [siberia, phan, scandinavian, russia's, balkan, leased, vladivostok, thessaloniki, bight, xinjiang, tripoli]
# 
# morning: [morning, evening, midnight, reached, afternoon, ten, stopped, six, along, later, late]
# 
# moon: [moon, sun, planet, dust, deep, dark, story, journey, sky, secret, earth]
# 
# nuclear: [nuclear, reactors, economic, uranium, industrial, world's, environmental, islamic, trade, reactor, prices]
# 
# ship: [ship, orders, destroyed, fired, ship's, officers, battles, remained, shot, stationed, boat]
# 
# job: [job, very, time, come, go, stay, much, become, well, take, thought]
# 
# wikipedia: [wikipedia, recent, wikipedia's, read, report, about, been, changes, message, arbitration, reports]
# 
# escape: [escape, kill, fight, escaping, thrown, sacrifice, trapped, crushed, fate, steal, turns]
# 
# outpost: [outpost, stationed, remained, retreating, retreat, barges, capturing, resulted, captured, battles, command]
# 
# car: [car, 2001, cars, interior, spa, market, loaned, virgin, jaguar, champ, cafe]
# 
# team: [team, champions, winning, win, tournament, arena, won, matches, compete, brands, scored]
# 
# hobbit: [hobbit, elves, gollum, sauron, bilbo, dwarves, middle-earth, frodo, orcs, gandalf, baggins]
# 
# forest: [forest, gippsland, valley, central, wood, pier, enfield, dover, north, green, gardens]
# 
# dragon: [dragon, revenge, madness, hercules, sword, magical, frees, disguised, ghoul, monster, ghosts]
# 
# imperial: [imperial, emperor's, raids, fled, morale, invade, emperor, arrived, invasion, fleet, jing]
# 
# metric: [metric, measurement, metrication, martinvl, bipm, kilogram, centaur, defacto, customary, indentured, thrasymedes]
# 
# empty: [empty, invalid, corresponding, specify, set, duplicate, template, syntax, inserted, template's, text]
# 
# dinosaurs: [dinosaurs, theropods, cretaceous, archosaurs, bipedal, archaeopteryx, fossils, mesozoic, theropod, fossil, dinosauria]
# 
# actor: [actor, comedian, producer, actress, animator, cartoonist, screenwriter, journalist, director, boxer, activist]
# 
# light: [light, bright, waves, smooth, glass, hole, pair, energy, sun, brightness, narrow]
# 
# jedi: [jedi, alien, wrath, adventure, sith, lego, ninja, voices, toon, phantom, aliens]
# 
# 
# ### Измененная метрика Минковского:
# 
# moon: [moon, planet, deep, magic, secret, yellow, opening, beautiful, volume, production, unknown]
# 
# job: [job, bring, stay, deal, rest, complete, completely, quickly, moment, thinking, decided]
# 
# wikipedia: [wikipedia, your, palatino, times, changes, message, wikidata, please, been, signpost, make]
# 
# outpost: [outpost, dusk, urging, calais, deserted, foreground, resisting, anchored, foe, cheering, deed]
# 
# car: [car, wins, driver, belgian, brazilian, cars, races, lotus, motor, hungarian, bull]
# 
# team: [team, won, previous, form, foundation, across, both, wmf, provide, four, end]
# 
# hobbit: [hobbit, elves, gollum, pippin, bilbo, sauron, orcs, springtime, cauldron, minerva, figaro]
# 
# forest: [forest, wood, garden, mars, greater, adams, zone, counties, banks, northwest, wolf]
# 
# dragon: [dragon, sword, monster, quest, chase, revealed, spy, magic, heroes, lion, journey]
# 
# imperial: [imperial, fleet, battles, invasion, launched, sons, refused, victory, headquarters, rising, officials]
# 
# metric: [metric, measurement, centaur, define, indicated, root, indicates, relation, martinvl, nervous, grounds]
# 
# empty: [empty, belong, authors, paste, replace, applies, contain, heading, titles, parent, translated]
# 
# dinosaurs: [dinosaurs, fossil, cretaceous, fossils, evolved, reptiles, jurassic, extinction, snakes, feathers, dinosaur]
# 
# light: [light, energy, earth, front, together, nature, ancient, outside, inside, lines, theory]
# 
# 
# 
# Несложно видеть, что в среднем косинусная мера выдает самый ожидаемый результат, хотя иногда ее опережает евклидова мера (особенно это видно при входном слове 'moscow'). Измененная мера Минковского, на мой взгляд, несмотря на кажущуюся ущербность, выдает иногда довольно нетривиальные, не совсем близкие, но все же связанные с исходным словом слова. Возможно, с помощью этой меры можно немного абстрагироваться от исходного слова. Например, тогда как косинусная мера связывала со словом 'dinosaurs' всякие виды динозавров, измененная мера Минковского среди близких слов нашла слово 'snake'. Мне кажется, это вполне неплох, если нужно уметь выйти за рамки данного контекста и подумать немного о более широких вещах. Конечно, в других случаях измененная мера Минковского дает какие-то странные ответы, но тем не менее.

# ## Skip-gram
# 
# Модель Skip-gram можно представить так: [входной вектор] -> [матрица представлений слов W$\cdot$d] -> [матрица представлений контекстов d$\cdot$W] -> [выходной вектор].
# 
# В качестве входного вектора будем подавать вектор, где все нули, а на месте, соответствующем слову, стоит 1; в качестве выходного вектора будем стремиться получить вектор из нулей и одной 1, стоящей на месте слова контекста.
# 
# В предположении, что перемножение матрицы и вектора происходит за линейное время от размера матрицы, сложность одной эпохи (проход по всей вики) составляет $O(W\cdot d \cdot T \cdot c)$, где $c$ - количество слов в контексте данного, $T$ - количество способов выбрать слово контекста, $W\cdot d$ - размер матрицы. Действительно, для каждого слова берется $c$ пар входных и выходных векторов, тогда за эпоху будет взято $T \cdot c$ пар входных и выходных векторов, для каждого вектора выполняется перемножение за $W\cdot d$. Стоит отметить, что для более точной оценки стоит добавить коэфф. примерно равный 4 (2 перемножения матриц, а затем обратных проход).
# 
# Данная оценка не предусматривает одну вещь: сложность нахождения значения функции потерь. Однако, можно считать, что коэфф. $k$ в формуле функции потерь очень мал. Я не знаю, как в tensorflow устроена NCE, но для нее существует возможность вычисления за время, сильно меньшее $W$, поскольку нам нужно брать только некоторые значения выходного вектора. В любом случае, считая, что сигмоида считается за $O(1)$, $k << W$, время вычисления функции потерь сильно меньше времени перемножения матриц. Поэтому наша оценка времени не изменится.

# Ну что же, время обучать модель. Поскольку такое ощущение, что keras написан на языке Кумир, у которого интерпретатор написан на Питоне, то пришлось использовать tensorflow.
# 
# Для начала считаем наш словарик.

# In[1]:

from __future__ import print_function
import numpy
numpy.random.seed(1337)  # for reproducibility

input = open('words-1.txt', 'r')
word_dict = dict()
s = input.readline()
while len(s) > 0:
    s = s.split()
    num = int(s[1])
    word_dict[s[0]] = num
    s = input.readline()
input.close()


# Теперь создадим модель. 
# 
# Я использовал размерность вектора $d=192$ для быстроты. Код почти полностью скопирован из базового примера tensorflow.

# In[4]:

import tensorflow as tf
import copy
import math

BATCH_SIZE = 128
EMB_SIZE = 192
graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([len(word_dict), EMB_SIZE], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([len(word_dict), EMB_SIZE],
                                 stddev=1.0 / math.sqrt(EMB_SIZE)))
        nce_biases = tf.Variable(tf.zeros([len(word_dict)]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=64,
                         num_classes=len(word_dict)))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    # Add variable initializer.
    init = tf.initialize_all_variables()


# Здесь я запускаю обучение. Выполняется 4 прохода по вики. Полученные матрицы сохраняются.

# In[ ]:

with tf.Session(graph=graph) as session:
    init.run()
    #print("Initialized")
    for loops in range(4):
        input = open('simplewiki-parsed3.txt', 'r')
        now_context = [-1 for i in range(9)]
        now_pred = numpy.zeros((BATCH_SIZE, 1))
        now_batch = numpy.zeros((BATCH_SIZE))
        now_ind = 0
        s = input.readline()
        c = 0
        prev_c = 0
        CNT = 0
        average_loss = 0
        while len(s) > 0:
            s = s.split()
            if len(s) > 0 and s[0] == "tttrrr":
                s = []
            for elem in s:
                for i in range(0, len(now_context) - 1):
                    now_context[i] = now_context[i + 1]
                now_context[-1] = word_dict[elem]
                if now_context[0] != -1:
                    for j in range(0, len(now_context)):
                        if j != len(now_context) // 2:
                            now_pred[now_ind, 0] = now_context[j]
                            now_batch[now_ind] = now_context[len(now_context) // 2]
                            now_ind += 1
                            if now_ind == BATCH_SIZE:
                                CNT += 1
                                feed_dict = {train_inputs: now_batch, train_labels: now_pred}
                                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                                now_ind = 0
                                average_loss += loss_val
            s = input.readline()
            c += 1
            if c // 50000 > prev_c:
                prev_c += 1
                #print(prev_c, end = '')
                average_loss /= 50000
                #print(": ", average_loss)
                average_loss = 0
        input.close()
    final_emb = (embeddings / tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))).eval()
    numpy.save('mas-Adagrad_norm', final_emb)
    final_emb2 = embeddings.eval()
    numpy.save('mas-Adagrad', final_emb2)
    final_emb = nce_weights.eval()
    numpy.save('__mas_weights', final_emb)
    final_emb2 = nce_biases.eval()
    numpy.save('__mas_biases', final_emb2)
    


# Мы сохранили наши матрицы, можно перейти к тестированию. Загружаем наш словарик и нашу матрицу представлений слов.

# In[6]:

import numpy
import copy

mas = numpy.load('mas-Adagrad.npy')
print(mas.shape)

input = open('words-1.txt', 'r')
word_dict = dict()
s = input.readline()
while len(s) > 0:
    s = s.split()
    num = int(s[1])
    word_dict[s[0]] = num
    s = input.readline()
print(len(word_dict))
input.close()


# Функция, которая по строке, представляющей из себя линейную комбинацию слов (см. пример), возвращает ближайшее слово. Фактически, это является расширенной версией функции $get\_nearest$. Также здесь убираются из списка ближайших слов слова, которые входили в линейную комбинацию.

# In[11]:

def get_nearest_linear_comb(s, n = 11, metric = "cos"):
    string = s.split()
    vect = mas[word_dict[string[0]]]
    for i in range(1, len(string), 2):
        vect2 = mas[word_dict[string[i + 1]]]
        if string[i] == '-':
            vect = vect - vect2
        elif string[i] == '+':
            vect = vect + vect2
        else:
            print('Error: wrong input string vector')
    min_d = [10000000000000000000000000 for i in range(n)]
    ans = ['' for i in range(len(min_d))]
    for word in word_dict:
        if metric == "cos":
            d = 2 - get_cos_dist_v(vect, word)
        else:
            d = get_dist_v(vect, word)
        if word in s:
            d = 1000000000000000000000000000000
        pst = False
        for p in range(0, len(min_d)):
            if not(pst) and d < min_d[p]:
                for j in range(len(min_d) - 1, p, -1):
                    min_d[j] = min_d[j - 1]
                    ans[j] = ans[j - 1]
                min_d[p] = d
                ans[p] = word
                pst = True
    print(' '.join(string) + ": [" + ', '.join(ans) + ']')


# In[212]:

get_nearest_linear_comb('putin + country', metric="cos")


# Косинусная метрика работает достаточно адекватно, тогда как евклидова работает довольно плохо. Одно из ее свойств: для подавляющего большинства слов она выдает слово %UNKNOWN в списке ближайших. Также в списке ближайших часто можно заметить "мусорные" слова: остатки разметки вики и т.д. Например, для слова 'cat' в евклидовой метрике ближайшие слова выглядят так:
# 
# [cats, dog, %UNKNOWN, bird, category, etc, face, pink, ref, local, family]
# 
# Видно, что тут помимо %UNKNOWN влезло минимум 3 мусорных слова: "etc", "ref", "category". Последние 2 из них, скорее всего, остались от разметки википедии.
# 
# По этой причине я приведу список близких слов только для косинусной меры.
# 
# putin: [vladimir, hosni, resigning, suleiman, ballmer, medvedev, andrei, stalin's, preparations, mikhail, khomeini] (очень странно на самом деле, список совсем другой, нежели в LSA)
# 
# moscow: [russian, petersburg, russia, soviet, sergei, ukrainian, oblast, ussr, kiev, belarus, berlin] (а вот тут список лучше, чем в LSA)
# 
# lord: [baron, temple, lords, god, king, earl, jesus, duke, sir, henry, christ]  (ну тут тоже чуть лучше, чем в LSA, как мне кажется)
# 
# computer: [rubbish, device, computing, logic, laptop, computers, hardware, graphics, electronics, mathematics, griffinofwales]    (rubbish, лол)
# 
# siberia: [russia, pacific, alaska, mongolia, europe, greenland, america, oceans, iceland, ussr, mountains] (ну, хоть не tripoli)
# 
# morning: [afternoon, evening, night, saturday, herald, day, sunday, late, friday, arrived, yesterday]
# 
# moon: [earth, mars, planet, jupiter, uranus, apollo, neptune, lunar, moons, spacecraft, sun]
# 
# nuclear: [atomic, reactor, chemical, weapons, energy, fusion, conservation, physics, power, radiation, solar]   (тут тоже лучше получилось)
# 
# ship: [ships, boat, sail, vessel, sank, sinks, sailing, navy, sunk, warship, sinking]    (и тут лучше)
# 
# job: [jobs, work, responsibility, idea, advice, skills, working, experience, training, care, him]  (гораздо лучше)
# 
# wikipedia: [welcome, changing, your, changes, english, simple, to, pages, thank, enwiki, project]
# 
# escape: [kill, escaped, enter, steal, survive, destroy, rescue, flee, catch, deliver, recover]
# 
# outpost: [regimes, uprisings, vicinity, deserted, plateaus, looting, outcast, casinos, sey, scarborough, systematically]
#   (вот здесь похуже)
# 
# car: [truck, bicycle, cars, automobile, driver, vehicle, motor, boat, racing, trucks, accident]   (а здесь получше)
# 
# team: [teams, league, championship, football, tournament, hockey, soccer, club, player, cup, play]
# 
# hobbit: [lego, penguin, shrek, monkey, chaucer, spider-man, middle-earth, pagos, sparrow, rogen, mcgowan]  (лего-хоббит, являющийся пингвином-обезьяной, который человек-паук в Средиземье :-) Ну хоть Средиземье есть)
# 
# forest: [forests, park, hills, pine, trees, mountains, tree, mountain, wildlife, reserve, habitat]
# 
# dragon: [monkey, pok, hulk, sword, panda, flower, tattoo, mon, kid, anime, boy]    (А-А-А-А!! Дракон-обезьяна-халк-панда-АНИМЕ)
# 
# imperial: [royal, navy, military, orders, marshal, japanese, empire, tokyo, emperor, naval, air]
# 
# metric: [measurement, limit, units, measures, measure, notation, measurements, max, newtons, yield, thus]
# 
# empty: [specified, parameter, otherwise, blank, caption, namespace, image, inside, string, invalid, itself]
# 
# dinosaurs: [mammals, reptiles, jurassic, cretaceous, birds, extinction, mesozoic, fossils, evolution, insects, creatures]
#   (и при чем тут млекопитающие?..)
# 
# actor: [actress, comedian, screenwriter, writer, singer, musician, film, composer, director, songwriter, television]  (чуть хуже, мне кажется)
# 
# light: [beam, waves, speed, sunlight, ultraviolet, bulb, sun, heat, shade, radiation, dust]
# 
# jedi: [clone, sith, dooku, ninja, obi-wan, kenobi, phantom, skywalker, darth, master, mutant]
# 
# september: [october, november, july, august, december, june, april, february, march, january, 11]    (ГОРАЗДО лучше)
# 
# star: [trek, planet, episode, trilogy, clone, stars, wars, sun, galaxy, dwarf, heroes]
# 
# nice: [interesting, okay, glad, fun, awesome, wonderful, ok, silly, big, btw, we'll]
# 
# language: [languages, dialect, words, word, alphabet, spoken, arabic, literature, writing, basic, speak]
# 
# street: [avenue, streets, wall, theatre, mall, corner, broadway, hotel, bus, square, 1929]

# #### Алгоритм определения столицы:
# moscow - russia + [country]. Пример: moscow - russia + poland: warsaw
# 
# #### Алгоритм нахождения множественного числа существительного:
# 
# [word] + countries - country. Примеры: 
# 
# city: [town, borough, city's, county, capital, district, metropolitan, province, river, mayor, state], но
# 
# city + countries - country: [cities, people, paris, town, areas, towns, 2011, borough, cases, geneva, river];
# 
# man: [woman, person, boy, man's, mask, men, soldier, girl, him, himself, guy], но
# 
# man + countries - country: [people, men, humans, woman, beings, camps, iron, victims, nations, societies, creatures]
# 
# Здесь важно сказать, почему именно country-countries. Дело в том, что у многих существительных единственное и множественное число лежат очень рядом. А у слова country среди близких слов нет countries. Аналогично были взяты слова для примеров: у слов man и city их формы множественного числа лежат не ближе всего, поэтому на их примере видно, что алгоритм работает. Впрочем, он также работает для слов, у которых множественное число лежит рядом с единственным.
# 
# #### Алгоритм нахождения превосходной степени прилагательного:
# 
# [word] + oldest - old. Примеры:
# 
# rich: [poor, z, young, fat, clay, dangerous, expensive, friends, loving, man, sugar], но
# 
# rich + oldest - old: [largest, world's, **richest**, smallest, longest, biggest, most, highest, earliest, important, strongest];
# 
# early: [late, beginning, mid, 19th, 18th, 1960s, earliest, 1940s, began, period, 17th], но
# 
# early + oldest - old: [largest, **earliest**, biggest, longest, world's, smallest, surviving, best-known, first, richest, late].
# 
# Тут видно, что возникают некоторые проблемы: например, упорно вылезают слова largest, biggest, smallest. Однако, это, видимо, связано со свойствами вики. Я пытался сделать более сложные комбинации для нахождения вектора превосходной степени (например, oldest - old + largest - large + [word]), но это только ухудшало ситуацию.
# 
# #### Алгоритм нахождения антонима:
# 
# [word] - elder + younger. Пример:
# 
# simple: [english, write, wiktionary, basic, me, writing, something, wikipedia, vocabulary, guidelines, paragraph], но
# 
# simple - elder + younger: [english, understand, me, administrators, **difficult**, simpler, articles, wiktionary, regular, write, **complex**]
# 
# На самом деле антонимы создавать фактически невозможно, поскольку очень часто антоним - одно из ближайших слов к данному слову. Это означает, что вектор разницы очень мал, что приводит к трудностям.
# 
# #### Алгоритм нахождения генерального директора компании:
# 
# Отсутствует. Ну то есть наверное это должно быть что-то типа [company] + zuckerberg - facebook, но поскольку ближайшее слово к zuckerberg - это koala; ближайшее к слову gates - это doors; слов yandex, vkontakte (vk) вообще нет в списке, как и владельцев/основателей google, то вариантов крайне мало, и ни один из них не работает.

# # GloVe

# Cначала загрузим словарик.

# In[12]:

import numpy
numpy.random.seed(1337)  # for reproducibility

input = open('words-1.txt', 'r')
word_dict = dict()
s = input.readline()
while len(s) > 0:
    s = s.split()
    num = int(s[1])
    word_dict[s[0]] = num
    s = input.readline()
input.close()


# Создадим нашу сеть. (Я почти дословно переписал код из примера реализации на github.)

# In[5]:

import tensorflow as tf
import copy
import math
from random import shuffle

BATCH_SIZE = 128
EMB_SIZE = 192
COUNT_MAX = 100
SCAL_FACTOR = 0.75
graph = tf.Graph()

with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    cooccurrence_count = tf.placeholder(tf.float32, shape=[BATCH_SIZE])
    count_max = tf.constant([COUNT_MAX], dtype=tf.float32)
    scal_factor = tf.constant([SCAL_FACTOR], dtype=tf.float32)
    
    embeddings = tf.Variable(tf.random_uniform([len(word_dict), EMB_SIZE], 1.0, -1.0))
    contexts = tf.Variable(tf.random_uniform([len(word_dict), EMB_SIZE], 1.0, -1.0))
    focal_biases = tf.Variable(tf.random_uniform([len(word_dict)], 1.0, -1.0))
    context_biases = tf.Variable(tf.random_uniform([len(word_dict)], 1.0, -1.0))

    focal_embedding = tf.nn.embedding_lookup([embeddings], train_inputs)
    context_embedding = tf.nn.embedding_lookup([contexts], train_labels)
    focal_bias = tf.nn.embedding_lookup([focal_biases], train_inputs)
    context_bias = tf.nn.embedding_lookup([context_biases], train_labels)
    
    magic_function = tf.minimum(1.0, tf.pow(tf.div(cooccurrence_count, count_max), scal_factor))
    embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1) 
    log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))
    distance_expr = tf.square(tf.add_n([embedding_product, focal_bias, context_bias, tf.neg(log_cooccurrences)]))

    single_losses = tf.mul(magic_function, distance_expr)
    total_loss = tf.reduce_sum(single_losses)
    optimizer = tf.train.AdagradOptimizer(0.1).minimize(total_loss)
        
    combined_embeddings = tf.add(embeddings, contexts)
    
    init = tf.initialize_all_variables()


# Создадим нужную нам матрицу встречаемости пар слов и запишем в файлы.

# In[8]:

WINDOW_SIZE = 9
now_context = ['' for i in range(WINDOW_SIZE)]
cooccurences_dict = dict()
input = open('simplewiki-parsed3.txt', 'r')
s = input.readline()
while len(s) > 0:
    s = s.split()
    if s[0] == 'tttrrr':
        s = []
    for elem in s:
        for j in range(0, len(now_context) - 1):
            now_context[j] = now_context[j+1]
        now_context[-1] = elem
        if now_context[0] != '':
            for j in range(len(now_context)):
                if j != len(now_context) // 2:
                    cur_word = word_dict[now_context[len(now_context) // 2]]
                    context_word = word_dict[now_context[j]]
                    if not((cur_word, context_word) in cooccurences_dict):
                        cooccurences_dict[(cur_word, context_word)] = 0
                    cooccurences_dict[(cur_word, context_word)] += 1
    s = input.readline()
print(len(cooccurences_dict))
input.close()


# In[10]:

output = open('context_words.txt', 'w')
for elem in cooccurences_dict:
    output.write(str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(cooccurences_dict[elem]) + '\n')
output.close()


# В реализации на github предлагалось выкидывать редкие пары слов. Спойлеры: на simplewiki результат оказался лучше, если ничего не выкидывать. Однако, я все же создал матрицы с выкинутыми редкими парами слов:

# In[11]:

output = open('context_words-8.txt', 'w')
for elem in cooccurences_dict:
    if cooccurences_dict[elem] >= 8:
        output.write(str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(cooccurences_dict[elem]) + '\n')
output.close()


# In[12]:

output = open('context_words-4.txt', 'w')
for elem in cooccurences_dict:
    if cooccurences_dict[elem] >= 4:
        output.write(str(elem[0]) + ' ' + str(elem[1]) + ' ' + str(cooccurences_dict[elem]) + '\n')
output.close()


# In[13]:

del cooccurences_dict


# Теперь из файла считаем нашу матрицу и распихаем по батчам.

# In[7]:

batches = list()
input = open('context_words.txt', 'r')
c = 0
prev_c = 0
num_batch = 0
while True:
    counts = [0 for _ in range(BATCH_SIZE)]
    i_s = [0 for _ in range(BATCH_SIZE)]
    j_s = [0 for _ in range(BATCH_SIZE)]
    is_good = True
    for i in range(BATCH_SIZE):
        s = input.readline()
        if not(len(s) > 0):
            is_good = False
            break
        s = s.split()
        ind = num_batch * BATCH_SIZE + i
        counts[i] = int(s[2])
        i_s[i] = int(s[0])
        j_s[i] = int(s[1])
        c += 1
        if c // 756626 > prev_c:
            prev_c += 1
            print(prev_c)
    if not(is_good):
        break
    batches.append((i_s, j_s, counts))
    num_batch += 1
input.close()


# In[8]:

print(len(batches) * BATCH_SIZE)


# Запустим обучение и запишем результаты в файл.

# In[9]:

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    for loops in range(15):
        shuffle(batches)
        num_batch = 0
        average_loss = 0
        c = 0
        prev_c = 0
        while num_batch < len(batches):
            feed_dict = {train_inputs: batches[num_batch][0],
                         train_labels: batches[num_batch][1],
                         cooccurrence_count: batches[num_batch][2]}
            num_batch += 1
            _, loss_val = session.run([optimizer, total_loss], feed_dict=feed_dict)
            average_loss += loss_val
            c += 1
            if c // 55000 > prev_c:
                prev_c += 1
                #print(loops, prev_c, end = '')
                average_loss /= 55000
                #print(": ", average_loss)
                average_loss = 0
    final_embeddings = combined_embeddings.eval()
    numpy.save('mas-GloVe-3', final_embeddings)
    final_embeddings = embeddings.eval()
    numpy.save('mas-GloVe-emb-3', final_embeddings)


# Загрузим результаты и протестируем.

# In[22]:

import numpy

mas = numpy.load('mas-GloVe-3.npy')
print(mas.shape)

input = open('words-1.txt', 'r')
word_dict = dict()
s = input.readline()
while len(s) > 0:
    s = s.split()
    num = int(s[1])
    word_dict[s[0]] = num
    s = input.readline()
print(len(word_dict))
input.close()


# In[23]:

get_nearest_linear_comb('king - man + woman', metric="cos")


# На самом деле несложно заметить, что GloVe работает гораздо нестабильнее word2vec. В принципе, причины этого интуитивно понятны, и скорее всего, на больших объемах информации GloVe все же лучше работает. С редкими же словами типа jedi тут вообще все плохо. 

# Теперь с помощью t-SNE сожмем размерности векторов.

# In[ ]:

from sklearn.manifold import TSNE

for i in range(7):
    tsne = TSNE(n_components=3, init='pca', n_iter=2000)
    low_dim_words = tsne.fit_transform(mas[i * 5000:(1+i) * 5000])
    numpy.save('3-dim-' + str(i), low_dim_words)
    print(i)
tsne = TSNE(n_components=3, init='pca', n_iter=2000)
low_dim_words = tsne.fit_transform(mas[35000:])
numpy.save('3-dim-7', low_dim_words)


# Загрузим теперь это в массив mas.

# In[13]:

mas = numpy.load('3-dim-0.npy')
for i in range(1, 8):
    mas_dim_tmp = numpy.load('3-dim-' + str(i) + '.npy')
    mas = numpy.vstack((mas, mas_dim_tmp))
    print(mas.shape)


# In[21]:

print(' king: ', mas[word_dict['king']], '; queen: ', mas[word_dict['queen']],
      '\n man: ', mas[word_dict['man']], '; woman: ', mas[word_dict['woman']],
      '\n boy: ', mas[word_dict['boy']], '; girl :', mas[word_dict['girl']], sep = '')
print(' big: ', mas[word_dict['big']], '; bigger :', mas[word_dict['bigger']], 
      '; biggest: ', mas[word_dict['biggest']],
      '\n old: ', mas[word_dict['old']], '; older :', mas[word_dict['older']], 
      '; oldest: ', mas[word_dict['oldest']], 
      '\n small: ', mas[word_dict['small']], '; smaller: ', mas[word_dict['smaller']], 
      '; smallest: ', mas[word_dict['smallest']], sep = '')


# Видно, что слова man, king, woman, boy, queen смотрят примерно в одном и том же направлении, как и old, older. Однако, со словами big, bigger, biggest, small, smaller, smallest очевидно, имеются какие-то проблемы. Аналогичную вещь можно сказать про girl.
# 
# Кстати, bigger, smaller направлены одинаково.

# Теперь касательно семплирования нулей матрицы встречаемости пар в GloVe. Внимательно посмотрев на формулу ошибки и статью авторов GloVe, можно увидеть, что добавление пар векторов с нулевой координатой в матрице в батч (увеличивая размер батча) вообще не повлияет на ошибку и на градиент. Это означает, что добавление в батч таких векторов никак не влияет на обучаемость. То есть батч размера $n+m$, в котором $n$ пар векторов с нулевой матричной координатой, равносилен батчу размера $m$. Таким образом, семплировать нули не имеется смысла. Более того, семплирование нулей очень замедлит обучение.

# Теперь давайте сравним модели на основе поиска аналогий. Для этого я выкачал и распарсил специальную таблицу. Каждая строка таблицы состоит из 4 слов $w_1, w_2, w_3, w_4$. Предполагается, что $w_4 = w_2 - w_1 + w_3$. На основе этого определяется, насколько способ превращения слов в векторы хорош. А именно, я смотрю ближайшие 10 слов к данной комбинации, выкидывая слова, входящие в нее. Если $w_4$ лежит в этом списке, то тест считается пройденным.

# Загрузим тест:

# In[25]:

words = list()
answers = list()
input = open('test_words_parsed.txt', 'r')
s = input.readline().split()
while len(s) == 4:
    words.append((s[0], s[1], s[2]))
    answers.append(s[3])
    s = input.readline().split()
input.close()
print(len(answers))


# Функция, возвращающая список ближайших к вектору слов:

# In[29]:

def return_nearest(s, n = 10, metric = 'cos'):
    string = s.split()
    vect = mas[word_dict[string[0]]]
    for i in range(1, len(string), 2):
        vect2 = mas[word_dict[string[i + 1]]]
        if string[i] == '-':
            vect = vect - vect2
        elif string[i] == '+':
            vect = vect + vect2
        else:
            print('Error: wrong input string vector')
    min_d = [10000000000000000000000000 for i in range(n)]
    ans = ['' for i in range(len(min_d))]
    for word in word_dict:
        if metric == "cos":
            d = 2 - get_cos_dist_v(vect, word)
        else:
            d = get_dist_v(vect, word)
        if word in s:
            d = 1000000000000000000000000000000
        pst = False
        for p in range(0, len(min_d)):
            if not(pst) and d < min_d[p]:
                for j in range(len(min_d) - 1, p, -1):
                    min_d[j] = min_d[j - 1]
                    ans[j] = ans[j - 1]
                min_d[p] = d
                ans[p] = word
                pst = True
    return ans


# В этой ячейке я по очереди загружал в память представления слов и тестировал:

# In[42]:

mas = numpy.load('mas-GloVe-3.npy')
count_true = 0
c = 0
prev_c = 0
for i in range(len(answers)):
    nearest = return_nearest(words[i][1] + ' - ' + words[i][0] + ' + ' + words[i][2], n=10)
    if answers[i] in nearest:
        count_true += 1
    c += 1
    if prev_c < c // 560:
        prev_c += 1
        print(prev_c * 4, '%', sep = '')
print(count_true / len(answers))


# ## Результаты
# **LSA** - 0.10529701908642505
# 
# **Skip-Gram** - 0.4973193223246837
# 
# **GloVe** - 0.2808635356351419

# ## И-и-и в нашем соревновании побеждает...
# 
# # ...Skip-Gram!!! 
# 
# ## *(громкие аплодисменты в зале, смущенный Skip-Gram выходит на сцену, играет торжественный гимн, одна из девушек кричит "я хочу от тебя эмбеддинги", GloVe в слезах выбегает из зала)*

# In[ ]:



