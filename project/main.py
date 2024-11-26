from math import *
import numpy as np
import matplotlib.pyplot as plt

# таблица Бутчера
dorman_prince = {}
dorman_prince.setdefault('a', [[0, 0, 0, 0, 0, 0],
                               [1 / 5, 0, 0, 0, 0, 0],
                               [3 / 40, 9 / 40, 0, 0, 0, 0],
                               [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
                               [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
                               [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0]])

dorman_prince.setdefault('b', [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])

# параметры модели
l = l1 = l2 = 1.2  # длины стержней
m1 = 7 # масса первого грузика
m2 = 11  # масса второго грузика
g = 9.81  # ускорение свободного падения (м/с)
mu = m2 / m1  # отношение масс
s_a1 = (radians(75))  # начальный угол наклона первого стержня
s_a2 = (radians(75))  # начальный угол наклона второго стержня


# производная от угла. здесь a1, a2 - текущий угол 1 и 2 шара, p1 и p2 - инерция тел,
# A1 и А2 - вспомогательные переменные
def Derivative_func(state):
    a1, a2, p1, p2 = state

    A1 = (p1 * p2 * sin(a1 - a2)) / (m1 * pow(l, 2) * (1 + mu * pow(sin(a1 - a2), 2)))
    A2 = ((pow(p1, 2) * mu - 2 * p1 * p2 * mu * cos(a1 - a2) +
           pow(p2, 2) * (1 + mu)) * sin(2 * (a1 - a2)) /
          (2 * m1 * pow(l, 2) * pow((1 + mu * pow(sin(a1 - a2), 2)), 2)))

    d_a1 = (p1 - p2 * cos(a1 - a2)) / (m1 * pow(l, 2) * (1 + mu * pow(sin(a1 - a2), 2)))
    d_a2 = ((p2 * (1 + mu)) - p1 * mu * cos(a1 - a2)) / m1 * pow(l, 2) * (1 + mu * pow(sin(a1 - a2), 2))
    d_p1 = -m1 * (1 + mu) * g * l * sin(a1) - A1 + A2
    d_p2 = -m1 * mu * g * l * sin(a2) + A1 - A2
    return np.array([d_a1, d_a2, d_p1, d_p2])


def DOPRI(state, h):
    a = dorman_prince['a']
    b = dorman_prince['b']
    size = len(a)
    k_array = np.zeros((size, len(state)))
    k_array[0] = Derivative_func(state)
    for i in range(1, size):
        for j in range(i):
            k_array[i] += h * a[i][j] * k_array[j]
        k_array[i] += state
        k_array[i] = Derivative_func(k_array[i])

    new_state = np.array(state)

    for i in range(size):
        new_state += h * b[i] * k_array[i]

    return new_state


def draw_grph(data, time):
    # y[:, 0] содержит все значения a1, а y[:, 1] - все значения a2
    plt.figure(figsize=(12, 8))

    colors = ['b', 'r']
    arr = ['a1', 'a2']

    # Накидываем шум
    noise_data = np.array(data)

    for i in range(len(noise_data)):
        noise_data[i] += np.random.normal(0.2, 0.2)

    np.savez('data.npz', data=data, noise_data=noise_data, time=time)

    plt.subplot(2, 1, 1)
    for i in range(2):  # Индексы 0 и 1 для a1 и a2
        plt.plot(time, [degrees(j) for j in noise_data[:, i]], colors[i], label=arr[i])

    plt.xlabel('Time')  # Подпись оси X
    plt.ylabel('Values of a1, a2')  # Подпись оси Y
    plt.legend()  # Легенда
    plt.grid(True)  # Сетка
    plt.title('График изменения a1 и a2 со временем с шумом')  # Заголовок графика

    # Построим графики для a1 и a2
    plt.subplot(2, 1, 2)  # указываем 2 строки, 1 столбец, выбираем второе место
    for i in range(2):  # Индексы 0 и 1 для a1 и a2
        plt.plot(time, [degrees(j) for j in data[:, i]], colors[i], label=arr[i])

    # Настройки графика
    plt.xlabel('Time')  # Подпись оси X
    plt.ylabel('Values of a1, a2')  # Подпись оси Y
    plt.legend()  # Легенда
    plt.grid(True)  # Сетка
    plt.title('График изменения a1 и a2 со временем')  # Заголовок графика
    plt.show()


# Пример вызова функции
# t = массив временных промежутков
# y = двумерный массив с a1, a2, p1, p2
def main():
    start = 0
    end = 10
    h = 0.005
    all_steps = np.arange(start, end, h)
    steps = int((end - start) / h)
    all_points = np.zeros((steps, 4))
    all_points[0] = [s_a1, s_a2, 0, 0]
    for i in range(1, steps):
        all_points[i] = DOPRI(all_points[i - 1], h)

    draw_grph(all_points, all_steps)


if __name__ == '__main__':
    main()
