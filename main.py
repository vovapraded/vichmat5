import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import sys
import math


def check_unique_x(x):
    if len(np.unique(x)) != len(x):
        print("Ошибка: значения x должны быть уникальными!")
        sys.exit(1)

def is_uniform_grid(x, tol=1e-9):
    h = x[1] - x[0]
    return np.all(np.abs(np.diff(x) - h) < tol)

# ==== Интерполяционные методы ====

def lagrange_interpolation(x, y, x_interp):
    result = 0
    for i in range(len(x)):
        term = y[i]
        for j in range(len(x)):
            if i != j:
                term *= (x_interp - x[j]) / (x[i] - x[j])
        result += term
    return result

def newton_divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])
    return coef

def newton_divided_eval(coef, x_data, x_interp):
    result = coef[-1]
    for i in range(len(coef) - 2, -1, -1):
        result = result * (x_interp - x_data[i]) + coef[i]
    return result

def newton_finite_diff(x, y, x_interp):
    h = x[1] - x[0]
    t = (x_interp - x[0]) / h
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    result = y[0]
    t_term = 1.0
    for i in range(1, n):
        t_term *= (t - i + 1) / i
        result += t_term * diff_table[0][i]
    return result

def stirling_interpolation_auto(xs, ys, x_interp):
    """
    Интерполяция по формуле Стирлинга (автоматическая сборка ряда).
    xs, ys - массивы узлов и значений
    x_interp - точка интерполяции
    """
    n = len(xs)
    if n % 2 == 0:
        raise ValueError("Стирлинг: только нечётное число узлов!")

    m = n // 2
    h = xs[1] - xs[0]
    t = (x_interp - xs[m]) / h

    # Строим таблицу конечных разностей
    diffs = [ys.copy()]
    for k in range(1, n):
        last = diffs[-1]
        diffs.append([last[i + 1] - last[i] for i in range(len(last) - 1)])

    # Сборка ряда
    result = ys[m]
    factorial = math.factorial

    # Для четных членов берем центральную разность
    # Для нечетных — среднее двух центральных разностей (симметрия)
    for k in range(1, n):
        # генерируем "многочлен" вида t(t^2-1)(t^2-4)... по правилу
        mult = 1
        for j in range(1, k + 1):
            if k % 2 == 1 and j == 1:
                mult *= t
            elif k % 2 == 0 and j == 1:
                mult *= t ** 2 - ((j - 1) ** 2)
            else:
                mult *= t ** 2 - ((j - 1) ** 2)
        if k % 2 == 1:
            idx1 = m - k // 2
            idx2 = m - k // 2 - 1
            if 0 <= idx1 < len(diffs[k]) and 0 <= idx2 < len(diffs[k]):
                central_diff = (diffs[k][idx1] + diffs[k][idx2]) / 2
            else:
                central_diff = 0
        else:
            idx = m - k // 2
            if 0 <= idx < len(diffs[k]):
                central_diff = diffs[k][idx]
            else:
                central_diff = 0
        result += mult / factorial(k) * central_diff
    return result


def bessel_interpolation_auto(xs, ys, x_interp):
    """
    Интерполяция по формуле Бесселя (автоматическая сборка ряда).
    xs, ys - массивы узлов и значений
    x_interp - точка интерполяции
    """
    n = len(xs)
    if n % 2 != 0:
        raise ValueError("Бессель: только чётное число узлов!")

    m = n // 2 - 1
    h = xs[1] - xs[0]
    x0 = (xs[m] + xs[m + 1]) / 2
    t = (x_interp - x0) / h

    # Строим таблицу конечных разностей
    diffs = [ys.copy()]
    for k in range(1, n):
        last = diffs[-1]
        diffs.append([last[i + 1] - last[i] for i in range(len(last) - 1)])

    # Сборка ряда
    result = (ys[m] + ys[m + 1]) / 2
    factorial = math.factorial

    # Суммируем все члены до конца ряда
    for k in range(1, n):
        # Для Бесселя четные и нечетные члены чередуются с коэффициентами t, t^2 - 1/4 и т.д.
        if k % 2 == 1:
            # нечетные члены — Δ^(2k-1)
            idx = m - (k // 2)
            if 0 <= idx < len(diffs[k]):
                diff = diffs[k][idx]
            else:
                diff = 0
            mult = t
            for j in range(1, k // 2 + 1):
                mult *= t ** 2 - (j - 0.5) ** 2
            term = mult / factorial(k) * diff
            result += term
        else:
            # четные члены — среднее Δ^(2k) двух центральных разностей
            idx1 = m - (k // 2) + 1
            idx2 = m - (k // 2)
            if 0 <= idx1 < len(diffs[k]) and 0 <= idx2 < len(diffs[k]):
                diff = (diffs[k][idx1] + diffs[k][idx2]) / 2
            else:
                diff = 0
            mult = 1
            for j in range(1, k // 2 + 1):
                mult *= t ** 2 - (j - 0.25) ** 2
            term = mult / factorial(k) * diff
            result += term
    return result


# ==== Ввод данных ====

def input_from_keyboard():
    n = int(input("Введите количество узлов: "))
    x = []
    y = []
    print("Введите значения x и y через пробел:")
    for _ in range(n):
        xi, yi = map(float, input().split())
        x.append(xi)
        y.append(yi)
    return np.array(x), np.array(y)

def input_from_file(path):
    try:
        data = np.loadtxt(path)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print("Ошибка чтения файла:", e)
        sys.exit(1)

def input_from_function(fx: Callable, a, b, n):
    x = np.linspace(a, b, n)
    y = fx(x)
    return x, y

# ==== Таблица разностей ====
def print_finite_diff_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    header = ["y"] + [f"Δ^{j}y" if j > 1 else "Δy" for j in range(1, n)]
    print("\nТаблица конечных разностей:")
    print(" ".join(f"{h:>10}" for h in header))

    for i in range(n):
        row = []
        for j in range(n - i):
            row.append(f"{table[i][j]:10.4f}")
        print(" ".join(row))
    return table
def print_divided_diff_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    # Печать
    header = ["x", "f[x]", *[" " * 3 + f"Δ^{j}f" for j in range(1, n)]]
    print("\nТаблица разделённых разностей Ньютона:")
    print(" ".join(f"{h:>13}" for h in header))
    for i in range(n):
        row = [f"{x[i]:13.6f}", f"{table[i, 0]:13.6f}"]
        for j in range(1, n - i):
            row.append(f"{table[i, j]:13.6f}")
        print(" ".join(row))
    return table

# ==== Визуализация всех методов ====

def plot_all_results(x, y, x_query, methods):
    x_dense = np.linspace(min(x), max(x), 300)
    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, label="Узлы", color="blue", zorder=3)
    plt.axvline(x_query, color="gray", linestyle="--", label=f"x = {x_query}", zorder=1)
    for name, fn, style, color in methods:
        plt.plot(x_dense, [fn(xi) for xi in x_dense], label=name, linestyle=style, color=color)
    plt.legend()
    plt.grid()
    plt.title("Интерполяция: сравнение методов")
    plt.tight_layout()
    plt.show()

# ==== Главный сценарий ====

def main():
    print("Выберите способ ввода данных:")
    print("1. Ввод с клавиатуры")
    print("2. Загрузка из файла")
    print("3. Вычисление по функции (например, sin(x), exp(x))")
    mode = input("Ваш выбор: ").strip()

    if mode == "1":
        x, y = input_from_keyboard()
    elif mode == "2":
        path = input("Введите путь к файлу: ").strip()
        x, y = input_from_file(path)
    elif mode == "3":
        print("Выберите функцию: 1 — sin(x), 2 — exp(x)")
        fmode = input("Ваш выбор: ").strip()
        fx = np.sin if fmode == "1" else np.exp
        a = float(input("Левая граница интервала: "))
        b = float(input("Правая граница интервала: "))
        n = int(input("Количество узлов: "))
        x, y = input_from_function(fx, a, b, n)
    else:
        print("Некорректный ввод")
        return

    check_unique_x(x)

    x_query = float(input("Введите значение x для интерполяции: "))
    print_divided_diff_table(x, y)

    methods = []
    # Всегда считаем Лагранжа и Ньютона по разделённым разностям
    lag_val = lagrange_interpolation(x, y, x_query)
    print(f"\n[Lagrange]   f({x_query}) = {lag_val:.6f}")
    methods.append(("Лагранж", lambda xx: lagrange_interpolation(x, y, xx), "dashed", "tab:purple"))

    newt_coef = newton_divided_diff(x, y)
    newt_val = newton_divided_eval(newt_coef, x, x_query)
    print(f"[Newton (divided)]   f({x_query}) = {newt_val:.6f}")
    methods.append(("Ньютон (разд.)", lambda xx: newton_divided_eval(newt_coef, x, xx), "solid", "tab:orange"))

    # Только для равномерной сетки — конечные разности, Стирлинг, Бессель
    if is_uniform_grid(x):
        print_finite_diff_table(x, y)
        newt_fin_val = newton_finite_diff(x, y, x_query)
        print(f"[Newton (finite)]    f({x_query}) = {newt_fin_val:.6f}")
        methods.append(("Ньютон (кон.р.)", lambda xx: newton_finite_diff(x, y, xx), "solid", "tab:green"))

        if len(x) % 2 == 1:
            try:
                stirling_val = stirling_interpolation_auto(x, y, x_query)
                print(f"[Stirling]   f({x_query}) = {stirling_val:.6f}")
                methods.append(("Стирлинг", lambda xx: stirling_interpolation_auto(x, y, xx), "dashdot", "tab:red"))
            except Exception as e:
                print(f"Ошибка Стирлинга: {e}")
        if len(x) % 2 == 0:
            try:
                bessel_val = bessel_interpolation_auto(x, y, x_query)
                print(f"[Bessel]     f({x_query}) = {bessel_val:.6f}")
                methods.append(("Бессель", lambda xx: bessel_interpolation_auto(x, y, xx), "dotted", "tab:brown"))
            except Exception as e:
                print(f"Ошибка Бесселя: {e}")
    else:
        print("ВНИМАНИЕ: Узлы неравномерные — методы по конечным разностям не применяются!")

    # Визуализация всех методов
    plot_all_results(x, y, x_query, methods)

if __name__ == "__main__":
    main()
