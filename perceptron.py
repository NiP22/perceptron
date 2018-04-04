import numpy as np
import random
'''''

import time

from functools import partial
from ipywidgets import interact, RadioButtons, IntSlider, FloatSlider, Dropdown, BoundedFloatText
from numpy.linalg import norm
'''
alphabet = {'а': 1, 'А': 1, 'б': 2, 'Б': 2, 'в': 3, 'В': 3, 'г': 4, 'Г': 4, 'д': 5, 'Д': 5, 'е': 6, 'Е': 6,
            'ё': 7, 'Ё': 7, 'ж': 8, 'Ж': 8, 'з': 9, 'З': 9, 'і': 10, 'І': 10, 'й': 11, 'Й': 11, 'к': 12, 'К': 12,
            'л': 13, 'Л': 13, 'м': 14, 'М': 14, 'н': 15, 'Н': 15, 'о': 16, 'О': 16, 'п': 17, 'П': 17, 'р': 18, 'Р': 18,
            'с': 19, 'С': 19, 'т': 20, 'Т': 20, 'у': 21, 'У': 21, 'ў': 22, 'Ў': 22, 'Ф': 23, 'ф': 23, 'х': 24, 'Х': 24,
            'ц': 25, 'Ц': 25, 'ч': 26, 'Ч': 26, 'ш': 27, 'Ш': 27, 'ы': 28, 'Ы': 28, 'ь': 29, 'Ь': 29, 'э': 30, 'Э': 30,
            'ю': 31, 'Ю': 31, 'я': 32, 'Я': 32}


def wordToVekt(word):
    mas = list(word)
    ans = []
    count = 0
    for i in mas:
        if i in alphabet:
            count += 1
            ans.append(alphabet[i])
        else:
            break
        if count > 14:
            break
    while count <= 14:
        count += 1
        ans.append(0)
    return ans


def listToStr(lis):
    ans = ""
    for i in lis:
        ans += (str(i) + ' ')
    return ans


outX = open("X.txt", "r")
outY = open("Y.txt", "w")

'''
f = open('train-bel.txt', 'r', encoding='utf-8')
line = f.read()
ans = list()
count = 0
for i in line.split():
    count += 1
    print(count)
    outX.write(listToStr(wordToVekt(i)) + ':')
    print(str(wordToVekt(i)))
'''


keys = outX.read()
mas = keys.split(sep=':')
a = np.zeros(shape=(len(mas) + 100000, 15))
X = np.array(list(map(lambda x: x.split(), mas)))
#ans[0] = list(map(lambda x: int(x), ans[0]))


Y = list()
for i in range(0, len(X)):
    Y.append(1)
    a[i] = np.array(list(map(lambda x: int(x), X[i])))
for i in range(len(X), len(X) + 99999):
    Y.append(0)
    a[i] = np.array([random.randint(1, 32) for i in range(15)])
print(a)

class Perceptron:
    def __init__(self, w, b):
        """
        Инициализируем наш объект - перцептрон.
        w - вектор весов размера (m, 1), где m - количество переменных
        b - число
        """

        self.w = np.array(w)
        self.b = b

    def forward_pass(self, single_input):
        """
        Метод рассчитывает ответ перцептрона при предъявлении одного примера
        single_input - вектор примера размера (m, 1).
        Метод возвращает число (0 или 1) или boolean (True/False)
        """

        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b

        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        """
        Метод рассчитывает ответ перцептрона при предъявлении набора примеров
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных
        Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
        (элементы вектора - boolean или целые числа (0 или 1))
        """
        return input_matrix.dot(self.w) + self.b > 0

    def train_on_single_example(self, example, y):
        """
        принимает вектор активации входов example формы (m, 1)
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """
        ans = self.forward_pass(example) > 0
        er = ans - y
        self.w -= er * (example.T[0])
        self.b -= er
        return ans

    def train_until_convergence(self, input_matrix, y, max_steps=1000):
        """
        input_matrix - матрица входов размера (n, m),
        y - вектор правильных ответов размера (n, 1) (y[i] - правильный ответ на пример input_matrix[i]),
        max_steps - максимальное количество шагов.
        Применяем train_on_single_example, пока не перестанем ошибаться или до умопомрачения.
        Константа max_steps - наше понимание того, что считать умопомрачением.
        """
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            print(self.w, i)
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)

                # int(True) = 1, int(False) = 0, так что можно не делать if

    def retW(self):
        print(self.w)


from sklearn.datasets import load_breast_cancer
import sklearn.model_selection as sk
cancer = load_breast_cancer()
#X_train, X_test, y_train, y_test = sk.train_test_split(a, np.array(Y), stratify=np.array(Y), random_state=66)
#print("пдлинна ", len(X_train[0]))
#print("первые пацаны", X_train[0:100])
#print(y_train)
Z = np.full(15, -1*random.random())
pc = Perceptron(np.array(Z), 0.1)
#print(np.array(X_train))
pc.train_until_convergence(a, Y)
pc.retW()

count = 0
countRight = 0
#for i in X_test:
 #   if(pc.forward_pass(np.array(i)) == np.array(y_test[count])):
    #    countRight += 1
  #  count += 1
#print(countRight/count)
