# Математическое моделирование колебаний двойного маятника методом Дормана-Принса 5 порядка

## Зависимости
Python3, numpy, matplotlib

## Описание
Для численного моделирования использовался метод Дормана-Принца 5 порядка.
На каждом шаге в качестве функции изменения угла отклонения использовалась следующие уравнения Гамильтона:

Здесь, a1, a2 - текущий угол 1 и 2 тел (второе тело прикреплено к первому), p1 и p2 - их инерция, A1 и А2 - вспомогательные переменные, l1 и l2 - длины стержней.

<p align="center">
    <img src="extra/img.png">

<p align="center">
    <img src="extra/img_1.png">

В учебной модели длины стержней взяты одинаковые (l1 = l2 = l), &mu; - отношение массы второго тела к массе первого, из-за чего система получает следующий вид:

<p align="center">
    <img src="extra/img_2.png">
    
<p align="center">
    <img src="extra/img_3.png">

Также к конечным результатам был применен шум, реализованный как случайное число в нормальном распределении, с мат ожиданием 0.2 и дисперсией 0.2.

## Пример результата

### Первый:

![img.png](extra/1.png)

![img_1.png](extra/2.png)

![noise_1.png](extra/noise_1.png)

### Второй:

![img.png](extra/9.png)

![img.png](extra/10.png)

![noise_2.png](extra/noise_2.png)


### Третий:

![img_2.png](extra/3.png)

![img_3.png](extra/4.png)

![noise_3.png](extra/noise_3.png)


### Четвертый:

![img.png](extra/5.png)

![img_1.png](extra/6.png)

![noise_4.png](extra/noise_4.png)


### Пятый:

![img.png](extra/7.png)

![img_1.png](extra/8.png)

![noise_5.png](extra/noise_5.png)


### Шестой:

![img.png](extra/11.png)

![img.png](extra/12.png)

![noise_6.png](extra/noise_6.png)


