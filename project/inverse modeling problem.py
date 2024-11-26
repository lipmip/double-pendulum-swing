# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt

# Загрузим данные из data.npz
data_file = np.load('data.npz')
data = data_file['data']  # Оригинальные данные
noise_data = data_file['noise_data']  # Данные с шумом
time = data_file['time']  # Временной массив

# Реализация фильтра Калмана
def kalman_filter(noisy_data, process_var, meas_var):
    """
    Реализация фильтра Калмана для одномерной переменной.
    :param noisy_data: массив наблюдений с шумом
    :param process_var: дисперсия процесса (Q)
    :param meas_var: дисперсия измерения (R)
    :return: восстановленные значения
    """
    n = len(noisy_data)
    x_est = np.zeros(n)  # Оценка состояния
    p_est = np.zeros(n)  # Оценка ковариации ошибки

    # Начальные значения
    x_est[0] = noisy_data[0]
    p_est[0] = 1.0

    for k in range(1, n):
        # Шаг предсказания
        x_pred = x_est[k - 1]
        p_pred = p_est[k - 1] + process_var

        # Шаг обновления
        kalman_gain = p_pred / (p_pred + meas_var)
        x_est[k] = x_pred + kalman_gain * (noisy_data[k] - x_pred)
        p_est[k] = (1 - kalman_gain) * p_pred

    return x_est

# Настройки фильтра Калмана
process_var = 1e-3  # Дисперсия процесса
meas_var = 0.04  # Дисперсия измерения

# Применим фильтр Калмана к углам отклонения
restored_a1 = kalman_filter(noise_data[:, 0], process_var, meas_var)
restored_a2 = kalman_filter(noise_data[:, 1], process_var, meas_var)

# Построим графики для сравнения
plt.figure(figsize=(12, 8))

# График для первого угла a1
plt.subplot(2, 1, 1)
plt.plot(time, np.degrees(data[:, 0]), 'g', label='Оригинальные a1')
plt.plot(time, np.degrees(noise_data[:, 0]), 'r', label='Зашумленные a1')
plt.plot(time, np.degrees(restored_a1), 'b', label='Восстановленные a1 (Калман)')
plt.xlabel('Время')
plt.ylabel('Угол (градусы)')
plt.title('Первый маятник (a1)')
plt.legend()
plt.grid()

# График для второго угла a2
plt.subplot(2, 1, 2)
plt.plot(time, np.degrees(data[:, 1]), 'g', label='Оригинальные a2')
plt.plot(time, np.degrees(noise_data[:, 1]), 'r', label='Зашумленные a2')
plt.plot(time, np.degrees(restored_a2), 'b', label='Восстановленные a2 (Калман)')
plt.xlabel('Время')
plt.ylabel('Угол (градусы)')
plt.title('Второй маятник (a2)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
