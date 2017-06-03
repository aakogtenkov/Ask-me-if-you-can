# Проект "Ask me"
Алексей Когтенков, 151 группа БПМИ ФКН ВШЭ

## О проекте

Довольно известным фактом является то, что многие компании (в частности, Google, Yandex и другие) занимаются проблемой распознавания речи, картинок и т.д. Одним из таких разделов в области распознавания является распознавание фраз, выделение из них сути и генерация ответа, если таковой известен системе. Примерами таких алгоритмов являются разнообразные чатботы. Также решение этой задачи позволяет быстро находить необходимую информацию в тексте (например, в википедии) машинным образом.

Не так давно появилась статья, в которой описывается способ реализации подобных программ при использовании одних только нейронных сетей. Это довольно новый подход, поскольку абсолютное большинство алгоритмов сейчас не используют в большом объеме нейронные сети, а содержат в себе решающие деревья, явные алгоритмы классификации и так далее. Реализация алгоритма, описанного в статье, - это и есть основная задача проекта. Попутно были решены также некоторые дополнительные задачи.

Ссылка на статью: https://arxiv.org/pdf/1506.07285.pdf

## Выполненная работа и разделы репозитория

При работе над проектом использованы и изучены полносвязные, конволюционные и рекуррентные нейронные сети, а также библиотеки языка Python: numpy, scipy, matplotlib, sklearn, tensorflow, keras.

Для разработки использовался язык Python3 и jupyter notebook.

Основной проект и все необходимые файлы находятся непосредственно в ветке master. Также в master находится папка labs с файлами выполненных лабораторных работ. Последние необходимы были исключительно для обучения, поэтому никакой документации и описания к ним не прилагается. Далее речь пойдет только об основном проекте.

## Краткое описание программы и важные сведения о её свойствах

Программа предоставляет пользователю простой интерфейс доступа к операциям создания, обучения и тестирования нейронной сети, которая для текста и вопроса по нему генерирует ответ на вопрос в виде _одного_ слова. Также доступны команды для подготовки данных для обучения/тестирования нейронной сети.

Приложение тестировалось на bAbI-task задании, и в текущей конфигурации оно заточено именно под него. А именно, при работе нейронной сети при кодировании слов используется one-hot кодирование, то есть скорость работы и количество потребляемой памяти линейно зависит от количества различных слов в текстах. В bAbI-task задании используется всего около 200 разных слов, однако не предполагается использование данного приложения в текущем виде на данных с большим количеством различных слов. Для использования на больших данных необходима модификация функций обработки текста и использование word2vec (реализация word2vec на данных из википедии была выполнена в одной из лабораторных).

## Системные требования

ОС: Linux, Windows*

Процессор: любой*

Оперативная память: минимум 4 Гб (bAbI-task)*

Дополнительные необходимые утилиты: интерпретатор Python3, библиотеки numpy и tensorflow для Python3.

\* Приложение разрабатывалось и проверялось на Ubuntu с 4-ядерным процессором 2.4 ГГц и 8 Гб оперативной памяти. Кроссплатформенность достигается за счет кроссплатформенности языка Python. Скорость процессора будет влиять на скорость работы программы, а количество оперативной памяти - на максимальный размер нейронной сети, которую можно создать и обучить.

## Туториал

Данный туториал показывает принцип взаимодействия пользователя с приложением на примере некоторых команд: подготовка данных, загрузка словаря и модели, тестирование модели.

В самом начале необходимо скачать код проекта askme.py из этого репозитория и файлы test.txt, dict.txt, test_network.index, test_network.meta, test_network.data-00000-of-00001, test_network.info. Перейдем в папку с кодом и файлами и запустим приложение с помощью консольной команды

    python3 askme.py
Появится приветственная фраза приложения. Прежде всего, нам нужно подготовить данные. Наши данные для тестирования записаны в обычном формате (см. "Поддерживаемые форматы данных") в файле test.txt. Используем команду prepare data (см. "Список команд").

    prepare data test.txt
В результате выполнения этой строки приложением будет создан файл 'test_prepared.txt' с подготовленными данными. Далее нам необходим словарь слов. Мы могли бы его создать с помощью соответствующей команды, однако в данном случае мы будем использовать уже готовый словарь, где содержатся не только слова из файла test_prepared.txt. Загрузим его командой

    load -d dict.txt
Теперь необходимо загрузить обученную версию из файла. В нашем случае она хранится под названием test_network:

    load -m ./test_network
Поскольку загрузка сети вызывает вначале построение графа вычислений, то время загрузки может быть до нескольких минут, в зависимости от мощности компьютера. После того, как программа выведет строку 'Done!', наша сеть полностью готова к работе. Осталось только запустить ее в режиме тестирования. Для этого введем команду

    test
после чего необходимо будет, как и при создании, ввести некоторые параметры тестирования. Мы хотим использовать размер батчей (пакетов) по умолчанию, который для тестирования равен 1, поэтому первый параметр пропустим. Второй параметр - это название файла с подготовленной выборкой для тестирования. Третий параметр выставляет флаг, будет ли для каждого вопроса выводиться ответ нейронной сети и верный ответ на стандартный поток вывода. Поскольку мы хотим визуально оценить качество работы, то выставляем значение yes. Четвертый параметр пропускаем, нажав Enter. В итоге получим такие строки:

    Please enter the following constants:

    Batch size (you can skip this step by typing 0): 0
    Name of file with test data: test_prepared.txt
    Output answers? (y/n): y
    End of string (zeros or EOF):
Под ними сразу же станут появляться строки вида:

    NN answer: garden,  real answer: bedroom
Первое слово - это ответ нейронной сети, второе - реальный ответ.

Поздравляем! Вы прошли данный туториал до конца и более чем на две трети освоили возможности пользовательского интерфейса и приложения в целом! Чтобы выйти из программы, можно ввести
    
       exit

Также в репозитории находятся файлы: big_test_prepared.txt - полные подготовленные данные для тестирования; gen_file.txt.7z - архив с подготовленными данными, на которых производилось обучение.

## Список команд

Программа представляет из себя заготовку небольшой библиотеки на языке Python3 с пользовательской оболочкой. Для пользователя, который хочет использовать реализованную нейронную сеть в чистом виде, предоставляется консольный интерфейс взаимодействия с помощью команд:

1) help \<command>. Эта команда позволяет посмотреть краткий мануал по использованию команды \<command>. 

2) prepare. У этой команды есть множество различных вариаций. Использование этой команды позволяет приводить исходные текстовые файлы к виду, который используется для обучения и тестирования нейронной сети.

      a) prepare data \<filename>. Эта команда приводит файл \<filename> с текстом в обычном формате (см. "Поддерживаемые форматы") к виду, который использует нейронная сеть, и сохраняет в файл, название которого генерируется автоматически исходя из названия \<filename>. ВАЖНО: для корректной работы этой команды название \<filename> должно заканчиваться на ".txt".
      
      b) prepare data \<filename> \<output>. Эта команда приводит файл \<filename> с текстом в обычном формате к виду, который использует нейронная сеть, и сохраняет в файл \<output>.
      
      c) prepare babidata \<filename>. Эта команда приводит файл \<filename> с текстом в bAbI-формате (см. "Поддерживаемые форматы") к виду, который использует нейронная сеть, и сохраняет в файл, название которого генерируется автоматически исходя из названия \<filename>. ВАЖНО: для корректной работы этой команды название \<filename> должно заканчиваться на ".txt".
      
      d) prepare babidata \<filename> \<output>. Эта команда приводит файл \<filename> с текстом в bAbI-формате к виду, который использует нейронная сеть, и сохраняет в файл \<output>.
      
      e) prepare dictionary \<filename>. Эта команда создает словарь по файлу \<filename> с подготовленными данными (т.е. сгенерированными с помощью команды prepare data или prepare babidata) и сохраняет его в файл с названием, которое генерируется исходя из названия файла \<filename>. После подготовки словаря он будет выведен на экран.
      
      f) prepare dictionary \<filename> \<output>. Эта команда создает словарь по файлу \<filename> с подготовленными данными (т.е. сгенерированными с помощью команды prepare data или prepare babidata) и сохраняет его в файл \<output>. После подготовки словаря он будет выведен на экран.

3) create model. Эта команда позволяет создать модель. После ввода этой команды появится контекстное интерактивное меню, в котором надо выбрать:
    
    Maximum number of sentences - максимальное количество предложений в тексте.
    
    Max size of dictionary - размер словаря, который мы хотим использовать для нейронной сети. По умолчанию это значение равно размеру текущего загруженного словаря.
    
    Typical size of hidden layers - характерный размер скрытых слоев нейронной сети.
    
    Number of passes through data in network - количество проходов по данным внутри сети (см. "Реализованная модель нейронной сети").
    
4) load. Эта команда позволяет загружать модели и словари. 

      a) load model \<path+name>. Эта команда позволяет загрузить модель с названием \<name> по пути \<path>, **причем даже есть модель хранится в той же папке, что и программа, путь должен быть все равно указан как ./, например: ./test_network**. Важно отметить, что одновременно приложение может хранить в оперативной памяти информацию только об одной модели. Загрузка модели затрет всю информацию о предыдущей модели. Также все параметры загружаемой модели должны совпадать с параметрами созданной модели. При использовании этой команды библиотека tensorflow может выдавать в консоль какие-то свои сообщения.
      
      b) load dictionary \<filename>. Эта команда позволяет загрузить словарь из файла \<filename>. Важно отметить, что одновременно приложение может хранить в оперативной памяти только один словарь. Загрузка словаря затрет предыдущий словарь. После загрузки словаря будет выведена его размер.
      
5) train. Эта команда позволяет обучать модель. После ввода этой команды появится контекстное интерактивное меню, в котором надо выбрать:
    
    Batch size - размер пакетов при обучении нейронной сети.
    
    Name of file with train data - название файла с подготовленными данными
    
    Number of epochs - количество эпох при обучении
    
    End of string (zeros or EOF) - тип замощения матриц текста в конце предложений. Если указан параметр EOF, то замощение будет специальным символов EOF, иначе - нулями.
         
    Frequency of saving model - частота сохрания нейронной сети в файл. Названия файлов генерируются исходя из файла с выборкой для обучения, частоты сохранения и номера эпохи.
    
    Важно сказать, что во время обучения никакие команды не доступны. Это сделано для ускорения работы. 
    
6) test. Эта команда позволяет тестировать модель. После ввода этой команды появится контекстное интерактивное меню, в котором надо выбрать:

    Batch size - размер пакета. Важно отметить, что если размер файла с выборкой для тестирования не кратен batch size, то последние (размер файла % batch size) примеры не будут протестированы.
    
    Name of file with test data - название файла с подготовленными данными
    
    Output answers? - введите 'y', чтобы для каждого примера выводился ответ нейронной сети и верный ответ.

    Ends of string (zeros or EOF) - тип замощения матриц текста в конце предложений. Если указан параметр EOF, то замощение будет специальным символов EOF, иначе - нулями.
    
    Важно сказать, что во время тестирования никакие команды не доступны. Это сделано для ускорения работы. 
    
7) save model \<path+name>. Эта команда позволяет сохранить текущую модель как модель по пути \<path> с названием \<name>. Важно: при сохранении модели создаются 3 файла, в которых хранится информация о ней. Удаление хотя бы одного из них приведет к невозможности восстановления модели.

Необходимо уточнить, что неверное использование команд в подавляющем большинстве случае будет приводить к ошибке в работе программы и ее закрытию.

## Файлы для обучения и тестирования нейронной сети, внутреннее представление

Для обучения и тестирования нейронной сети в моем приложении используются текстовые файлы определенного вида. Эти файлы делятся на 2 типа: основной файл и словарь. Для корректной работы нейронной сети в режиме обучения и тестирования необходимы оба этих файла. Основной файл всегда состоит из так называемых абзацев, каждый абзац имеет вид

    sentence1
    sentence2
    ...
    sentencen
    QUESTION
    question
    ANSWER
    answer
где sentence, question и answer - это строки вида

    word1 word2 ... wordm EOF
где word - это слово.
Пример такого абзаца:

    brian is black EOF
    emily is yellow EOF
    QUESTION
    what color is emily EOF
    ANSWER
    yellow EOF
Абзацы должны разделяться пустыми строками.
Файл со словарем имеет вид

    ind1 word1
    ind2 word2
    ...
    indk wordk
где word - это слово, ind - его номер в словаре. Оба файла генерируются приложением с помощью пользовательских команд, поэтому пользователь может обойтись без прямого взаимодействия с их содержимым. Однако важно отметить, что все слова в обоих файлах не содержат заглавных букв.

## Поддерживаемые форматы данных

Для более удобного пользования приложением была реализована функция преобразования текстовых данных для обучения/тестирования в используемый нейронной сетью формат (см. "Файлы для обучения и тестирования нейронной сети"). Эта функция поддерживает 2 формата записи данных:

1) bAbI-task формат. В этом формате данные состоят из так называемых абзацев. Каждый абзац состоит из строк вида

        N Word1 word2 ... wordn.
    
      или

        N Word1 word2 ... wordn?   answer n1 n2 n3 ... nk
    
    Здесь word - это слово; N - номер предложения в абзаце; n1, n2, ..., nk - номера предложений, из которых следует ответ; answer - это слово-ответ на строку-вопрос. Стоит отметить, что чисел n1, n2, ..., nk может и не быть.

    Пример куска текста в bAbI-task формате:

        12 Sandra got the milk there.
        13 Mary went to the bedroom.
        14 What is Sandra carrying? 	milk	12
        15 Sandra moved to the garden.
        16 Daniel journeyed to the hallway.
        17 What is Sandra carrying? 	milk	12
        1 Sandra went back to the bedroom.
        2 Daniel got the football there.

2) Обычный (интуитивный) формат. В этом формате данные состоят из абзацев. Каждый абзац имеет вид

        Word1 word2 word3 ... wordn.
    
      или
        
        Word1 word2 ... wordn? answer n1 n2 ... nk
    
      Здесь word - это слово; n1, n2, ..., nk - номера предложений, из которых следует ответ; answer - это слово-ответ на строку-вопрос. Стоит отметить, что чисел n1, n2, ..., nk может и не быть.

## Реализованная модель нейронной сети

Подробное описание нейронной сети содержится в файле Net description.pdf. Это связано с тем, что github не поддерживает формулы в readme. Ниже приводится краткое описание.

Нейронная сеть состоит из нескольких модулей, связанных между собой. 

**Модуль ввода:**

Вначале входной текст кодируется как последовательность векторов, каждый вектор кодирует одно слово текста или символ окончания предложения. В моей реализации используется one-hot кодирование. Эта последовательность подается на вход рекуррентной сети GRU, которая преобразовывает эти векторы в новую последовательность. Из этой последовательности выбирается подпоследовательность векторов, соответствующих символам конца предложения. Эта последовательность является выходом модуля ввода.

Аналогичным образом преобразуется вопрос. Поскольку вопрос состоит только из одного предложения, то на выходе будет только один вектор.

**Модуль эпизодической памяти:**

Описание этого модуля стоит смотреть в исходной статье, поскольку он весьма сложен для описания и включает в себя формулы. Единственным отличием моей реализации от предложенной в статье является то, что в векторе z (см. статью) нету последних двух координат, которые получаются применением матрицы W_b к некоторым векторам. Это сделано для ускорения обучения, поскольку кроме как в этом месте, матрица W_b нигде не используется.

В модуле эпизодической памяти используются параметры:

NUM_PASSES (Number of passes through data in network) - это количество проходов по входным данным в этом модуле;

HIDDEN_SIZE (Typical size of hidden layers) - типичный размер всех скрытых слоев, размеры которых не определяются входными матрицами данных;

NUM_SENTENCES (Maximum number of sentences) - максимальное количество предложений в тексте. В этом модуле присутствует итерация по предложениям, и в моей реализации эту константу требуется задавать.

Результатом работы модуля эпизодической памяти является вектор.

**Модуль ответа:**

Этот модуль генерирует по вопросу q (см. модуль ввода) и результату работы m модуля эпизодической памяти ответ. В моей реализации ответом всегда является одно слово, поэтому ответ A вычисляется по формуле

A = softmax(W_a * GRU([0, q], m)), где W_a - это некоторая матрица; 0 - это вектор длины HIDDEN_SIZE, который по сути нужен только если мы хотим генерировать ответ длины больше одного слова; [] - это конкатенация; GRU(x, y) - это применение матриц рекуррентной сети GRU для входа x и текущего состояния сети y. Дописав несколько строк кода в модуль, можно добиться генерации последовательности векторов в качестве ответа (см. статью).

В моей реализации в качестве выхода модуля ответа получается один вектор, ближайшее слово к которому (в one-hot кодировании) считается ответом. Обучение производится при помощи функции кросс-энтропии.

Помимо основных 3 файлов, которые создает tensorflow, я также добавил файл с расширением .info, в котором хранятся константы созданной сети.

## Результаты

Для обучения и тестирования нейронной сети были выбраны 16 из 20 заданий bAbI-task'a: все, кроме 3, 5, 8, 19 (некоторые из последних либо содержали ответы на вопросы из более чем одного слова, либо были неудобны для обучения в силу длины текстов). На выбранных 16 заданиях была обучена нейронная сеть, способная отвечать на все эти 16 типов вопросов. Для обучение понадобилось около 30 эпох (проходов по всем данным). Качество сети на отложенной выборке составило 90% при минимальном качестве на одном типе вопросов более 70%. 

Однако оказалось, что на реальных текстах и вопросах (которые составляются человеком на основе заданий bAbI-task'a) сеть показывает гораздо худший результат (качество около 60-70%). Связано это, скорее всего, с тем, что она переобучается: специфика сгенерированных компьютером заданий bAbI очень сильна. 

тоит отметить, что авторы исходной статьи обучали почти такую же сеть на каждом из заданий по отдельности, а не одновременно на всех. Таким образом, данный проект стал доказательством тому, что сеть из статьи вполне может научиться отвечать одновременно на разные типы вопросов.

## Заключение

В рамках проекта была создана, обучена и протестирована нейронная сеть, генерирующая ответ по тексту и вопросу. Также была сделана обертка в виде консольного интерфейса для людей, которые незнакомы с Питоном или не хотят напрямую писать код для работы с реализованным классом сети и другими функциями. 

Результаты показали, что это направление весьма перспективно, и дальнейшие улучшения результатов возможны. Например, можно применять не one-hot кодирование для слов, а кодирование с помощью word2vec. Такое изменение даст возможность обучать сеть на статьях википедии, сообщениях в соц. сетях и так далее. 

Обученную сеть можно применять для создания виртуального помощника, - программы, которая будет быстро отвечать на вопросы, основываясь на информации из интернета; чат-ботов; средств быстрого поиска и т.д.
