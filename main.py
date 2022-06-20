# lab2
# Вароиант  - 18
# Функция - -x^3 +5.67x^2 - 7.12x + 1.34
# Корни: 0.228, 1.486, 3.956
# Метооды - 3(Метод Ньютона), 5(Метод простой итерации), 2(Метод хорд) / Првый, левый, центральный
# Методы в программе: 1(Метод половинного деления), 5(Метод простой итерации), 6(Метод Ньютона)

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import sys

ks = 0

class Equation(object):
    def __init__(self, symbol, expression, text_expression, mil, mar, ep):
        self.symbol = symbol
        self.expression = expression
        self.text_expression = text_expression
        self.mil = mil
        self.mar = mar
        self.ep = ep


def print_error():
    print("Неверно введено значение, попробуйте еще раз!")


def get_eps_len(epsilon):
    return str(int(len(str(epsilon)) - 1))

def build_graph(equation, left, right, epsilon, x_solve):
    xx = np.arange((left - abs(right) * 0.1) * 1.1, ((right + abs(left) * 0.1) * 1.1), epsilon)
    y = [equation.expression.subs(equation.symbol, ii) for ii in xx]
    fig = plt.subplots()
    ax = plt.gca()
    plt.plot(xx, y)
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(y=0, color='g')
    ax.axvline(x=0, color='g')
    plt.plot(x_solve, [equation.expression.subs(equation.symbol, x_i) for x_i in x_solve], "ro")

    plt.grid()
    plt.show()


def build_graph_for_3(f, dots_x, dots_y):
    fig, ax = plt.subplots()

    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.axhline(y=0, color='g')
    ax.axvline(x=0, color='g')

    lf = lambda x, y: x ** 2 + y ** 2 - 4  # todo
    lg = lambda x, y: y - 3 * (x ** 2) + ks  # todo
    # lg = lambda x, y: 5*x - 2*y + 5
    x_range = np.arange(f.mil, f.mar, f.ep)
    y_range = np.arange(f.mil, f.mar, f.ep)
    xx, yy = np.meshgrid(x_range, y_range)
    f_f = lf(xx, yy)
    g_g = lg(xx, yy)

    plt.contour(xx, yy, g_g, [0])
    plt.contour(xx, yy, f_f, [0])

    for i in range(0, len(dots_x)):
        plt.plot(dots_x[i], dots_y[i], "ro")

    plt.grid()
    plt.show()


def variant_choose():
    while True:
        try:
            inp = int(input())
            if inp == 1:
                return True
            elif inp == 2:
                return False
            else:
                raise ValueError
        except ValueError:
            print_error()


def choose_input():
    print("Выберите способ ввода данных: ")
    print("1: С клавиатруы")
    print("2: Из файла (input.txt)")
    return variant_choose()


def choose_write():
    print("Выберите способ вывода данных: ")
    print("1: В консоль")
    print("2: В файл (output.txt)")
    return variant_choose()


def print_function(eq1, eq2):
    print("Система:")
    print(eq1)
    print(eq2)


def keyboard_input_all(stream_in, stream_out):
    method = keyboard_input_choose_method()
    if method == 3:
        _x, _y = symbols("x y")
        eq = Equation(_x, _x ** 2 + _y ** 2 - 4, "x^2 + y^2 = 4", -10, 10, 0.01)  # todo
        stream_in = Equation(_y, _y - 3*_x**2 + ks, "y = 3*x^2", eq.mil, eq.mar, eq.ep)  # todo
        print_function(eq.text_expression, stream_in.text_expression)
        build_graph_for_3(eq, [], [])
        left = keyboard_input_fridge("Введите значение x начального приближения от", eq.mil, eq.mar)
        right = keyboard_input_fridge("Введите значение y начального приближения от", eq.mil, eq.mar)
    else:
        eq = keyboard_input_choose_function()
        build_graph(eq, eq.mil, eq.mar, 0.01, [])
        while True:
            left = keyboard_input_fridge("Введите левую границу интервала от", eq.mil, eq.mar)
            right = keyboard_input_fridge("Введите правую границу интервала от", eq.mil, eq.mar)
            if left >= right:
                print("Ты что шизик??? КАК ТЫ СМЕЕШЬ ПИСАТЬ ТАКОЕ?")
            else:
                break

    epsilon = keyboard_input_fridge("Введите погрешность от", 0, 1)

    info = funcs[method - 1](eq, left, right, epsilon, stream_out, stream_in)

    if info is None:
        return None
    if method == 3:
        build_graph_for_3(eq, info[0], info[1])
    else:
        build_graph(eq, left, right, epsilon, info)


def keyboard_input_choose_method():
    print("Выберите метод решения: ")
    print("1: Метод половинного деления для нелинейного уравнения")
    print("2: Метод простой инетрации для нелинейго уравнения")
    print("3: Метод Ньютона для системы нелинейных уравнений")
    while True:
        method = input_choose_method(sys.stdin, 3)
        if method is None:
            print_error()
            continue
        else:
            return method


def keyboard_input_choose_function():
    print("Выберите функцию")
    i = 0
    for eq in equations:
        i += 1
        print(i, ": ", eq.text_expression)
    while True:
        fun = input_choose_method(sys.stdin, len(equations))
        if fun is None:
            print_error()
        else:
            return equations[fun - 1]


def keyboard_input_fridge(text, v_min, v_max):
    print(text, v_min, "до", v_max, ": ")
    while True:
        fridge = input_float(sys.stdin, v_min, v_max)
        if fridge is None:
            print_error()
        else:
            return fridge

        ###########################


def file_input_all(stream_input, stream_out):
    method = input_choose_method(stream_input, 3)
    if method is None:
        return None
    _x, _y = symbols("x y")
    if method == 3:
        eq = Equation(_x, _x ** 2 + _y ** 2 - 4, "x^2 + y^2 = 4", -3, 3, 0.01)
    else:
        function = input_choose_method(stream_input, 3)
        if function is None:
            return None
        eq = equations[function - 1]

    left = input_float(stream_input, eq.mil, eq.mar)
    if left is None:
        return None
    right = input_float(stream_input, eq.mil, eq.mar)
    if right is None or left >= right:
        return None
    eps = input_float(stream_input, 0, 1)
    if eps is None:
        return None
    if method == 3:
        stream_input = Equation(_y, _y - 3 * _x ** 2, "y = 3*x^2", eq.mil, eq.mar, eq.ep)

    info = funcs[method - 1](eq, left, right, eps, stream_out, stream_input)
    if info is None:
        return None
    if method == 3:
        build_graph_for_3(eq, info[0], info[1])
    else:
        build_graph(eq, left, right, eps, info)
    return True


def input_choose_method(text_input, count):
    try:
        i = int(text_input.readline())
        if (i < 1) or (i > count):
            raise ValueError
        else:
            return i
    except ValueError:
        return None


def input_float(text_input, v_min, v_max):
    try:
        i = float(text_input.readline())
        if (i < v_min) or (i > v_max):
            raise ValueError
        else:
            return i
    except ValueError:
        return None


def half_div(equation, left, right, epsilon, stream_output, stream_input):
    lengt = get_eps_len(epsilon)
    float_str = "{0:." + lengt + "f}"
    n = 1
    if equation.expression.subs(equation.symbol, left) * equation.expression.subs(equation.symbol, right) > 0:
        stream_output.write("Ошбика введенных границ!\n")
        return None
    while True:
        x_n = float((left + right) / 2)
        stream_output.write("----------------\n")

        stream_output.write("Итерация номер: ")
        stream_output.write(str(n))
        stream_output.write("\n")

        stream_output.write("Левая граница: ")
        stream_output.write(str(float_str.format(left)))
        stream_output.write("\n")

        stream_output.write("Правая граница: ")
        stream_output.write(str(float_str.format(right)))
        stream_output.write("\n")

        stream_output.write("x_n = ")
        stream_output.write(float_str.format(x_n))
        stream_output.write("\n")

        stream_output.write("f(x_n) = ")
        stream_output.write(str(float_str.format(equation.expression.subs(equation.symbol, x_n))))
        stream_output.write("\n")

        stream_output.write("----------------\n")

        if equation.expression.subs(equation.symbol, left) * equation.expression.subs(equation.symbol, x_n) > 0:
            left = x_n
        else:
            right = x_n
        n += 1
        if abs(left - right) <= epsilon or abs(equation.expression.subs(equation.symbol, x_n)) < epsilon:
            break

    stream_output.write("Кол-во итераций: ")
    stream_output.write(str(n))
    stream_output.write("\n")

    stream_output.write("x = ")
    stream_output.write(str(float_str.format(x_n)))
    stream_output.write("\n")

    stream_output.write("f(x) = ")
    stream_output.write(str(float_str.format(equation.expression.subs(equation.symbol, x_n))))
    stream_output.write("\n")
    return [x_n]


def simple_iter(equation, left, right, epsilon, stream_output, stream_input):
    lengt = get_eps_len(epsilon)
    float_str = "{0:." + lengt + "f}"
    if stream_input == "system":
        print("Введите начально приближение от", left, "до", right)
        x_0 = 0
        while True:
            x_0 = input_float(sys.stdin, left, right)
            if x_0 is None:
                print_error()
                continue
            break
    else:
        x_0 = input_float(stream_input, left, right)
        if x_0 is None:
            return None
    dif_f = diff(equation.expression)

    try:
        res_max = maximum(dif_f, equation.symbol, Interval(left, right))
    except NotImplementedError:
        return None

    __lambda = float(float_str.format(- 1 / res_max))
    phi = equation.symbol + __lambda * equation.expression
    if abs(diff(phi).subs(equation.symbol, x_0)) >= 1:
        stream_output.write("Не прошло провреки на возможность поиска\n")
        return None

    n = 0
    stream_output.write("lambda is: " + str(__lambda) + "\n")
    stream_output.write("Phi(x) = " + str(phi) + "\n")
    prev = x_0
    while True:
        n += 1
        now = phi.subs(equation.symbol, prev)
        stream_output.write("------------\n")
        stream_output.write("Номер итерации: " + str(n) + "\n")
        stream_output.write("prev: " + str(float_str.format(prev)) + "\n")
        stream_output.write("now: " + str(float_str.format(now)) + "\n")
        stream_output.write("f(x): " + str(float_str.format(equation.expression.subs(equation.symbol, now))) + "\n")
        stream_output.write("x: " + str(float_str.format(now)) + "\n")
        stream_output.write("|x_i-1 - x_i| = " + str(float_str.format(abs(prev - now))) + "\n")

        if abs(prev - now) <= epsilon or abs(equation.expression.subs(equation.symbol, now)) <= epsilon:
            stream_output.write("~~~~LAST~~~~")
            return [now]
        prev = now


def newton(eq1, x0, y0, epsilon, stream_output, eq2):
    lengt = get_eps_len(epsilon)
    float_str = "{0:." + lengt + "f}"

    xx = eq1.symbol
    yy = eq2.symbol
    temp = eq1
    eq11 = diff(eq1.expression, xx)
    eq12 = diff(eq1.expression, yy)
    eq21 = diff(eq2.expression, xx)
    eq22 = diff(eq2.expression, yy)

    # joc = eq11 * eq22 - eq12 * eq21

    dx = Symbol("dx")
    dy = Symbol("dy")
    # eq1 = joc * dx + eq1
    # eq2 = joc * dy + eq2
    eq1 = eq11 * dx + eq12 * dy + eq1.expression
    eq2 = eq21 * dx + eq22 * dy + eq2.expression
    n = 0
    while True:
        # print(eq1)
        # print(eq2)
        feq1 = eq1.subs(xx, x0).subs(yy, y0)
        feq2 = eq2.subs(xx, x0).subs(yy, y0)
        # print(feq1)
        # print(feq2)

        dx0, dy0 = solve_linear(feq1, feq2, dx, dy)  # dx and dy

        # sol = linsolve([feq1, feq2], dx, dy)
        # (x0, y0) = next(iter(sol))

        x_n = dx0 + x0
        y_n = dy0 + y0

        n += 1

        stream_output.write("--------------\n")
        stream_output.write("Итерация номер: " + str(n) + "\n")
        stream_output.write("x0 = " + str(float_str.format(x0)) + "\n")
        stream_output.write("y0 = " + str(float_str.format(y0)) + "\n")
        stream_output.write("dx = " + str(float_str.format(dx0)) + "\n")
        stream_output.write("dy = " + str(float_str.format(dy0)) + "\n")

        if (abs(x_n - x0) <= epsilon) or (abs(y_n - y0) <= epsilon):
            stream_output.write("--------------\n")
            stream_output.write("Ответ получен!" + "\n")
            stream_output.write("Кол-во итераций: " + str(n) + "\n")
            stream_output.write(
                "Полученная точка: " + str(float_str.format(x_n)) + " " + str(float_str.format(y_n)) + "\n")
            return [x_n], [y_n]

        x0 = x_n
        y0 = y_n

        stream_output.write(
            "new x0 - " + str(float_str.format(x0)) + "\n" + "new y0 - " + str(float_str.format(y0)) + "\n")


def solve_linear(eq1, eq2, s1, s2):
    c1 = -eq1.subs(s1, 0).subs(s2, 0)
    c2 = -eq2.subs(s1, 0).subs(s2, 0)
    a1 = eq1.subs(s1, 1).subs(s2, 0) + c1
    a2 = eq1.subs(s1, 0).subs(s2, 1) + c1
    b1 = eq2.subs(s1, 1).subs(s2, 0) + c2
    b2 = eq2.subs(s1, 0).subs(s2, 1) + c2
    print(c1, c2, a1, a2, b1, b2)
    a = np.array([[float(a1), float(a2)], [float(b1), float(b2)]])
    b = np.array([float(c1), float(c2)])
    # j = float(a1) * float(b2) - float(a2) * float(b1)
    # j1 = float(c1) * float(b2) - float(c2) * float(a2)
    # j2 = float(a1) * float(c2) - float(c1) * float(b1)
    return np.linalg.solve(a, b)


def define_equations():
    x = Symbol('x')
    # equations.append(
    #   Equation(x, x ** 3 - 1 * x + 4, "-x^3 +5.67x^2 - 7.12x + 1.34", -3.5, 4.5, 3))
    equations.append(
        # Equation(x, -x ** 3 + 5.67 * x ** 2 - 7.12 * x + 1.34, "-x^3 +5.67x^2 - 7.12x + 1.34", -0.5, 4.5, 3))
        Equation(x, x**3 - x + 4, "-x^3 - x + 4", -2, 1, 3))
    equations.append(
        Equation(x, -1.38 * x ** 3 - 5.42 * x ** 2 + 2.57 * x + 10.95, "-1.38x^3 - 5.42x^2 + 2.57x + 10.95", -5, 2.6,
                 3))
    equations.append(Equation(x, x * sin(x) + 5 / 2 * cos(x) - 2, "xsin(x) + 5/2 cos(x) - 2", -10, 10, 2))


if __name__ == "__main__":
    equations = list()
    define_equations()
    c_in = choose_input()
    funcs = np.array([half_div, simple_iter, newton])

    if choose_write():
        f_out = sys.stdout
    else:
        f_out = open("output.txt", "w+", encoding="utf-8")

    if c_in:
        keyboard_input_all("system", f_out)
    else:
        f_in = open("input.txt", "r+")
        if file_input_all(f_in, f_out) is None:
            print("Произошла ошибка чтения из файла попробуйте еще раз")
        f_in.close()

    if f_out != sys.stdout:
        f_out.close()
