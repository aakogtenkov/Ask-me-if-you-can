import tensorflow as tf
import numpy
import random

def tolower(s):
    for i in range(len(s)):
        if s[i] >= 'A' and s[i] <= 'Z':
            s = s[:i] + chr(ord(s[i]) - ord('A') + ord('a')) + s[i + 1:]
    return s



def preparing_data(filename = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt', fo=None, flag='babi'):
    input = open(filename, 'r')
    output = None
    if fo == None:
        output = open(filename[:-4] + '_prepared.txt', 'w')
    else:
        output = open(fo, 'w')
    s = input.readline()
    cur_text = ""
    while len(s) > 0:
        s = tolower(s)
        s = s.replace('.', '')
        s = s.replace(',', '')
        s = s.replace('!', '')
        if s.find('?') != -1:
            s = s.split('?')
            left = s[0].split()
            if flag == 'babi':
                left = s[0].split()[1:]
            right = s[1].split()
            for i in range(len(right)):
                if len(right[i]) > 0 and '9' >= right[i][0] >= '0':
                    right[i] = ''
            output.write(cur_text + 'QUESTION\n' + ' '.join(left) + ' EOF \n' + 'ANSWER\n' + ' '.join(right) + ' EOF \n\n')
            if flag != 'babi':
                cur_text = '';
        else:
            if flag == 'babi':
                if s.split()[0] == '1':
                    cur_text = ""
                s = ' '.join(s.split()[1:]) + ' EOF \n'
            else:
                s = ' '.join(s.split()[0:]) + ' EOF \n'
            if s == ' EOF \n':
                s = ''
            cur_text += s
        s = input.readline()
    input.close()
    output.close()


def creating_dictionary(filename='tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train_prepared.txt', foname=None):
    ii = filename.rfind('/')
    dictionary = None
    if foname == None:
        fo = filename[:ii + 1] + 'dict_' + filename[ii + 1:]
        fo = fo.replace('_prepared.txt', '.txt')
        dictionary = open(fo, 'w')
    else:
        dictionary = open(foname, 'w')
    d = dict()
    input = open(filename, 'r')
    s = input.readline()
    while len(s) > 0:
        if len(s.split()) > 1:
            words = s.split()
            for word in words:
                if not(word in d):
                    d[word] = len(d)
        s = input.readline()
    for word in d:
        print(word, d[word])
        dictionary.write(str(d[word]) + ' ' + word + '\n')
    input.close()
    dictionary.close()


def load_dictionary(filename='tasks_1-20_v1-2/en-10k/dict_q0_14691112131415161718207102-55s59t.txt'):
    input = open(filename, 'r')
    d = dict()
    s = input.readline()
    while len(s) > 0:
        s = s.split()
        d[s[1]] = int(s[0])
        s = input.readline()
    #print(d)
    print('Dict. length:', len(d))
    input.close()
    return d


class Model:
    def __init__(self, 
                 NUM_SENTENCES = 58,
                 INPUT_SIZE = 100,
                 HIDDEN_SIZE = 100,
                 NUM_PASSES = 5):
        self.graph = tf.Graph()
        self.NUM_SENTENCES = NUM_SENTENCES
        self.INPUT_SIZE = INPUT_SIZE
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NUM_PASSES = NUM_PASSES
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE], name='i') #batch_size; max_words; input_size
            self.train_question = tf.placeholder(tf.float32, shape=[None, None, INPUT_SIZE], name='q') #batch_size; len_question; input_size
            self.lookup = tf.placeholder(tf.float32, shape=[None, NUM_SENTENCES, None]) #batch_size; num_sentences; max_words
            with tf.variable_scope("question_module"):
                _, q_state = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(num_units=HIDDEN_SIZE), 
                                               self.train_question, dtype = tf.float32)
            with tf.variable_scope("input_module"):
                input_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(num_units=HIDDEN_SIZE), 
                                                   self.train_inputs, dtype = tf.float32)
                input_output = tf.matmul(self.lookup, input_output)

            BATCH_SIZE = tf.shape(self.train_inputs)[0]
            with tf.variable_scope("attention_mechanism"):
                m_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
                am_cell = tf.contrib.rnn.GRUCell(num_units=HIDDEN_SIZE) 
                m_state = m_cell.zero_state(BATCH_SIZE, tf.float32)
                cur_ep = tf.Variable(tf.zeros(HIDDEN_SIZE))
                cur_m = q_state
                for i in range(NUM_PASSES):
                    am_state = am_cell.zero_state(BATCH_SIZE, tf.float32)
                    prev_state = am_cell.zero_state(BATCH_SIZE, tf.float32)
                    for t in range(NUM_SENTENCES):
                        c = input_output[:, t, :]
                        z = tf.concat([c, cur_m, q_state, c * q_state, c * cur_m, 
                                       tf.abs(c - q_state), tf.abs(c - cur_m)], 1)

                        g1 = tf.tanh(tf.layers.dense(z, HIDDEN_SIZE, name='dense1', reuse=bool(t + i)))
                        g = tf.sigmoid(tf.layers.dense(g1, HIDDEN_SIZE, name='dense2', reuse=bool(t + i)))
                        if t > 0 or i > 0:
                            tf.get_variable_scope().reuse_variables()
                        _, am_state = am_cell(c, am_state)
                        am_state = tf.add(g * am_state, (1-g)*prev_state)
                        prev_state = am_state
                    cur_ep = am_state
                    _, m_state = m_cell(cur_ep, m_state)

            W_a = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_SIZE]))
            loss = 0
            self.train_answer = tf.placeholder(tf.int32, shape=[None, INPUT_SIZE], name='a')
            with tf.variable_scope("answer_module"):
                a_cell = tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
                a_state = m_state
                prev_predict = a_cell.zero_state(BATCH_SIZE, tf.float32)
                _, a_state = a_cell(tf.concat([prev_predict, q_state], 1), a_state)
                self.prev_predict = (tf.transpose(tf.matmul(W_a, a_state, transpose_b=True)))
                loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.train_answer, self.prev_predict))
            self.accuracy = tf.losses.mean_squared_error(self.train_answer, tf.nn.softmax(self.prev_predict))
            self.optimizer = tf.train.AdamOptimizer().minimize(loss)

            init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init)
    
    def __del__(self):
        self.session.close()
    
    def restore(self, filename=None):
        if filename == None:
            print("Requirement: filename")
            return 1
        self.saver.restore(self.session, filename)
        return 0
    
    def learn(self, epochs=10, batch_size=100, filename=None, freq_saving=4, d=None, flag='learn', end_type='zeros'):
        if d == None:
            print('Requirement: dictionary')
            return 1
        if filename == None:
            print('Requirement: filename')
            return 2
        freq_saving = max(freq_saving, 1)
        BATCH_SIZE = batch_size
        questions_txt = [''] * BATCH_SIZE
        train_text_txt = [''] * BATCH_SIZE
        max_words = 0
        question_len = 0
        train_answers = numpy.zeros((BATCH_SIZE, self.INPUT_SIZE), dtype=numpy.float32)
        cur_s = 0
        input = None
        for ep in range(epochs):
            c = 0
            if flag == 'learn':
                print("Epoch: ", ep)
            s = ''
            input = open(filename, 'r')
            s = input.readline()
            stage = 't'
            cur_text = ""
            average_loss = 0
            while len(s) > 0:
                v = s.split()
                if len(v) > 1 and stage != 'a':
                    cur_text += s
                elif len(v) > 0 and v[0] == "QUESTION":
                    stage = 'q'
                    train_text_txt[cur_s] = cur_text
                    max_words = max(max_words, len(cur_text.split()))
                    cur_text = ""
                elif len(v) > 0 and v[0] == "ANSWER":
                    stage = 'a'
                    questions_txt[cur_s] = cur_text
                    question_len = max(question_len, len(cur_text.split()))
                    cur_text = ""
                elif stage == 'a' and len(v) > 0:
                    cur_text = s.split()
                    train_answers[cur_s, d[cur_text[0]]] = 1
                    cur_text = ""
                    stage = 't'
                    cur_s += 1
                if cur_s == BATCH_SIZE:
                    train_text = numpy.zeros((BATCH_SIZE, max_words, self.INPUT_SIZE), dtype=numpy.float32)
                    train_questions = numpy.zeros((BATCH_SIZE, question_len, self.INPUT_SIZE), dtype=numpy.float32)
                    lookups = numpy.zeros((BATCH_SIZE, self.NUM_SENTENCES, max_words), dtype=numpy.float16)
                    for _ in range(BATCH_SIZE):
                        current_text = train_text_txt[_].split()
                        lookup_ind = 0
                        lkps = 0
                        for __ in range(len(current_text)):
                            train_text[_, __, d[current_text[__]]] = 1
                            if current_text[__] == 'EOF':
                                lookups[_, lookup_ind, lkps] = 1
                                lookup_ind += 1
                            lkps += 1
                        current_question = questions_txt[_].split()
                        for __ in range(len(current_question)):
                            train_questions[_, __, d[current_question[__]]] = 1
                        if end_type == 'EOF':
                            for __ in range(len(current_question), question_len):
                                train_questions[_, __, d['EOF']] = 1
                    feed_dict = {self.train_inputs: train_text, self.train_question: train_questions, 
                                     self.train_answer: train_answers, self.lookup: lookups}
                    if flag == 'learn':
                        answer = self.prev_predict.eval(session=self.session, feed_dict=feed_dict)
                        for i in range(BATCH_SIZE):
                            average_loss += int(numpy.argmax(answer[i]) == numpy.argmax(train_answers[i]))
                        _, loss_val = self.session.run([self.optimizer, self.accuracy], feed_dict=feed_dict)
                    else:
                        answer = self.prev_predict.eval(session=self.session, feed_dict=feed_dict)
                        for i in range(BATCH_SIZE):
                            average_loss += int(numpy.argmax(answer[i]) == numpy.argmax(train_answers[i]))
                            if flag == 'predict_write':
                                ind_ans = numpy.argmax(answer[i])
                                for word in d:
                                    if d[word] == ind_ans:
                                        print("NN answer:", word, end = ', ')
                                ind_true_ans = numpy.argmax(train_answers[i])
                                for word in d:
                                    if d[word] == ind_true_ans:
                                        print(" real answer:", word)
                    cur_s = 0
                    train_answers = numpy.zeros((BATCH_SIZE, self.INPUT_SIZE), dtype=numpy.int32)
                    max_words = 0
                    question_len = 0
                    c += BATCH_SIZE
                s = input.readline()
            print('Average loss:', average_loss / c)
            input.close()
            if flag == 'learn' and ep % freq_saving == 0:
                self.save(filename + '_' + str(ep // freq_saving))
                #self.saver.save(self.session, filename + '_' + str(ep // freq_saving))
        return 0
    
    def predict(self, batch_size=100, filename=None, d=None, flag='predict', end_type='zeros'):
        return self.learn(epochs=1, batch_size=batch_size, filename=filename, d=d, flag=flag)
    
    def save(self, filename=None):
        if filename == None:
            print('Requirement: file to save')
            return 1
        self.saver.save(self.session, filename)
        output = open(filename + '.info', 'w')
        output.write(str(self.NUM_SENTENCES) + '\n' + str(self.INPUT_SIZE) 
                     + '\n' + str(self.HIDDEN_SIZE) + '\n' + str(self.NUM_PASSES))
        output.close()
        print('The last save is in file ' + filename)


def main_func():
    stage = -1
    print('Hello! This is a program for "ask me" project.')
    s = input()
    s = s.split()
    model = None
    d = None
    file_gen_answer = list()
    file_gen_question = list()
    file_gen_text = list()
    while True:
        if len(s) == 1 and s[0] == 'exit':
            return 0
        elif (len(s) == 1 or len(s) == 2) and s[0] == 'help':
            if len(s) == 1:
                print('Commands: help, prepare, create, load, save, train, test.')
                print('Print "help <command>" for details.')
            elif s[1] == 'help':
                print('"help help"?? You are really mad, man. :-)')
            elif s[1] == 'create':
                print('Command "create" is used for creating model.')
                print('"create model" or "create -m" will start creating model, you have to follow instructions after it.')
            elif s[1] == 'prepare':
                print('Command "prepare" is used for preparing data and dictionary before training.')
                print('"prepare data <filename>" will create a file with prepared data from file <filename>.')
                print('"prepare data <filename> <output>" will create a file with prepared data from file <filename> with name <output>')
                print('"prepare babidata <filename>" and "prepare babidata <filename> <output>" do the same things, '
                      'but parser is specific for bAbI data. Note that every string in bAbI set starts with a number of string')
                print('"prepare dictionary <filename>" will create a dictionary from file <filename>.'
                      ' Note that this file must be with prepared data')
                print('"prepare dictionary <filename> <output>" does the same, but dictionary will be written to file <output>')
            elif s[1] == 'save':
                print('Command "save" is used for saving models.')
                print('"save model <filename>" or "save -m <filename>" will save your current model to file <filename>')
            elif s[1] == 'load':
                print('Command "load" is used for loading models and dictionaries.')
                print('"load model <name>" or "load -m <name>" will load model with name <name>.')
                print('"load dictionary <filename>" or "load -d <filename>" will load dictionary from file <filename>.')
            elif s[1] == 'train':
                print('Command "train" is used for training current model.')
                print('"train" will start training model, you have to follow instructions after it.')
            elif s[1] == 'test':
                print('Command "test" is used for testing model and predicting.')
                print('"test" will start testing, you have to follow instructions after it.')
        elif len(s) == 3 and s[0] == 'load':
            if (s[1] == 'model' or s[1] == '-m'):
                info = open(s[2] + '.info', 'r')
                NUM_SENTENCES, INPUT_SIZE, HIDDEN_SIZE, NUM_PASSES = map(int, info.readlines())
                info.close()
                model = Model(NUM_SENTENCES=NUM_SENTENCES,
                              INPUT_SIZE=INPUT_SIZE,
                              HIDDEN_SIZE=HIDDEN_SIZE,
                              NUM_PASSES=NUM_PASSES)
                model.restore(s[2])
                print('Done!')
            elif s[1] == 'dictionary' or s[1] == '-d':
                d = load_dictionary(s[2])
        elif len(s) == 3 and s[0] == 'save':
            if (s[1] == 'model' or s[1] == '-m') and model != None:
                model.save(s[2])
        elif (len(s) == 3 or len(s) == 4) and s[0] == 'prepare':
            if s[1] == 'data':
                print('NOT BABI PARSER IS USED')
                if len(s) == 3:
                    preparing_data(s[2], flag='notbabi')
                else:
                    preparing_data(s[2], s[3], flag='notbabi')
                print('Done!')
            elif s[1] == 'babidata':
                print('BABI PARSER IS USED')
                if len(s) == 3:
                    preparing_data(s[2])
                else:
                    preparing_data(s[2], s[3])
                print('Done!')
            elif s[1] == 'dictionary':
                if len(s) == 3:
                    creating_dictionary(s[2])
                else:
                    creating_dictionary(s[2], s[3])
        elif len(s) == 2 and s[0] == 'create':
            if s[1] == 'model' or s[1] == '-m':
                print('Please enter the following constants:\n')
                print('Maximum number of sentences: ', end='')
                NUM_SENTENCES = int(input())
                print('Max size of dictionary (you can skip this step by typing 0 if you already have loaded dictionary): ', end='')
                INPUT_SIZE = int(input())
                while (True):
                    if INPUT_SIZE <= 0 and d != None:
                        INPUT_SIZE = len(d)
                        break
                    if INPUT_SIZE > 0:
                        break
                    print('Wrong value. Please try again.')
                    print('Max size of dictionary (you can skip this step by typing 0, but only if you already have loaded dictionary): ', end='')
                    INPUT_SIZE = int(input())
                print('Typical size of hidden layers (you can skip this step by typing 0): ', end='')
                HIDDEN_SIZE = int(input())
                if HIDDEN_SIZE <= 0:
                    HIDDEN_SIZE = 100
                print('Number of passes through data in network (you can skip this step by typing 0): ', end='')
                NUM_PASSES = int(input())
                if NUM_PASSES <= 0:
                    NUM_PASSES = 5
                print('Creating model. Please, wait...')
                model = Model(NUM_SENTENCES=NUM_SENTENCES,
                              INPUT_SIZE=INPUT_SIZE,
                              HIDDEN_SIZE=HIDDEN_SIZE,
                              NUM_PASSES=NUM_PASSES)
                print('Done!')
        elif len(s) == 1 and s[0] == 'test':
            print('Please enter the following constants:\n')
            print('Batch size (you can skip this step by typing 0): ', end='')
            batch_size = max(int(input()), 1)
            print('Name of file with test data: ', end='')
            filename = input().rstrip()
            print('Output answers? (y/n): ', end='')
            flag = input().rstrip()
            if flag == 'y':
                flag = 'predict_write'
            else:
                flag = 'predict'
            print('Ends of string (zeros or EOF): ', end='')
            end_type = input().rstrip()
            model.predict(batch_size=batch_size, filename=filename, d=d, flag=flag, end_type=end_type)
            print('Done!')
        elif len(s) == 1 and s[0] == 'train':
            print('Please enter the following constants:\n')
            print('Batch size: ', end='')
            batch_size = max(int(input()), 1)
            print('Name of file with train data: ', end='')
            filename = input().rstrip()
            print('Number of epochs: ', end='')
            epochs = max(int(input()), 0)
            print('End of string (zeros or EOF): ', end='')
            end_type = input().rstrip()
            print('Frequency of saving model (you can skip this step by typing 0): ', end='')
            freq_saving = int(input())
            if freq_saving <= 0:
                freq_saving = 4
            print('Training...')
            model.learn(epochs=epochs, batch_size=batch_size, filename=filename, 
                        freq_saving=freq_saving, d=d, end_type=end_type)
            print('Done!')
        elif len(s) == 1 and s[0] == 'gen':
            if len(file_gen_answer) == 0:
                print('Please enter the filename.')
                s = input().rstrip()
                cur_text = ''
                file_gen_answer = list()
                file_gen_question = list()
                file_gen_text = list()
                fin = open(s, 'r')
                s = fin.readline()
                while len(s) > 0:
                    if s == '\n' and len(cur_text) > 0:
                        cur_text = cur_text.split('QUESTION\n')
                        file_gen_text.append(cur_text[0].replace(' EOF', '.'))
                        cur_text = cur_text[1].split('ANSWER\n')
                        file_gen_question.append(cur_text[0].replace(' EOF', '?'))
                        file_gen_answer.append(cur_text[1].replace(' EOF', ''))
                        cur_text = ''
                    else:
                        cur_text += s
                    s = fin.readline()
                fin.close()
                print('\n', len(file_gen_text), "examples")
            randn = random.randint(0, len(file_gen_text) - 1)
            print('\n', file_gen_text[randn].rstrip(), sep='')
            print('QUESTION')
            print(file_gen_question[randn].rstrip())
            print('ANSWER')
            print(file_gen_answer[randn].rstrip(), '\n')
            s = input().rstrip()
            while s != 'e' and s != 'close' and s != 'exit' and s != 'c':
                if s == 'a' or s == 'accept':
                    output = open('tmp.tmp', 'w')
                    output.write(tolower(file_gen_text[randn]).replace('.', ' EOF').rstrip() + '\nQUESTION\n' + 
                                 tolower(file_gen_question[randn]).replace('?', ' EOF').rstrip() + '\nANSWER\n' + 
                                 tolower(file_gen_answer[randn]).rstrip() + ' EOF\n')
                    output.close()
                    print()
                    if model != None:
                        model.predict(batch_size=1, filename='tmp.tmp', d=d, flag='predict_write')
                        print()
                        randn = random.randint(0, len(file_gen_text) - 1)
                    else:
                        print('ERROR: no model loaded.')
                elif s == 'chq':
                    s = input()
                    file_gen_question[randn] = s
                elif s == 'cha':
                    s = input().rstrip()
                    file_gen_answer[randn] = s
                elif s == 'cht':
                    text = file_gen_text[randn].split('\n')
                    for n_string in range(len(text)):
                        print(n_string, text[n_string])
                    print(file_gen_question[randn].rstrip(), end = ' ')
                    print('Answer:', file_gen_answer[randn].rstrip(), '\n')
                    s = input().rstrip()
                    while s != 'ok':
                        s = s.split()
                        if len(s) == 2:
                            pos = max(0, int(s[1]))
                            pos = min(len(text), pos)
                            if s[0] == 'rm' and pos < len(text):
                                text.pop(pos)
                            elif s[0] == 'rm':
                                print('Index out of range')
                            elif (s[0] == 'add' or s[0] == 'ins') and pos <= len(text):
                                s = input().rstrip()
                                text.insert(pos, s)
                            elif s[0] == 'add' or s[0] == 'ins':
                                print('Index out of range')
                            else:
                                print('Unknown command')
                        for n_string in range(len(text)):
                            print(n_string, text[n_string])
                        print(file_gen_question[randn].rstrip(), end = ' ')
                        print('Answer:', file_gen_answer[randn].rstrip(), '\n')
                        s = input().rstrip()
                    file_gen_text[randn] = '\n'.join(text) + '\n'
                else:
                    randn = random.randint(0, len(file_gen_text) - 1)
                print('\n', file_gen_text[randn].rstrip(), sep='')
                print('QUESTION')
                print(file_gen_question[randn].rstrip())
                print('ANSWER')
                print(file_gen_answer[randn].rstrip(), '\n')
                s = input().rstrip()
        else:
            print('Unknown command.')
        s = input().split()


main_func()

