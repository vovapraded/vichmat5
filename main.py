import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import sys

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

def stirling_interpolation(x, y, x_interp):
    n = len(x)
    if n % 2 == 0:
        raise ValueError("Стирлинг работает только для нечётного числа узлов!")
    h = x[1] - x[0]
    m = n // 2
    t = (x_interp - x[m]) / h
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    res = diff_table[m, 0]
    fact = 1
    tt = t
    for k in range(1, m+1):
        d1 = (diff_table[m - k, 2 * k - 1] + diff_table[m - k + 1, 2 * k - 1]) / 2
        d2 = diff_table[m - k, 2 * k]
        fact *= (t ** 2 - (k - 1) ** 2) / (2 * k)
        res += tt * d1 + fact * d2
        tt *= (t ** 2 - k ** 2) / (2 * k + 1)
    return res

def bessel_interpolation(x, y, x_interp):
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Бессель работает только для чётного числа узлов!")
    h = x[1] - x[0]
    m = n // 2 - 1
    t = (x_interp - (x[m] + x[m+1]) / 2) / h
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
    res = (diff_table[m, 0] + diff_table[m+1, 0]) / 2
    res += t * diff_table[m, 1]
    res += (t**2 - 0.25) * (diff_table[m, 2] + diff_table[m+1, 2]) / 2 / 2
    res += t * (t**2 - 1) * (diff_table[m, 3] + diff_table[m+1, 3]) / 2 / 6
    res += (t**2 - 1) * (t**2 - 2.25) * (diff_table[m, 4] + diff_table[m+1, 4]) / 2 / 24
    return res

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
    print_finite_diff_table(x, y)

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
        newt_fin_val = newton_finite_diff(x, y, x_query)
        print(f"[Newton (finite)]    f({x_query}) = {newt_fin_val:.6f}")
        methods.append(("Ньютон (кон.р.)", lambda xx: newton_finite_diff(x, y, xx), "solid", "tab:green"))

        if len(x) % 2 == 1:
            try:
                stirling_val = stirling_interpolation(x, y, x_query)
                print(f"[Stirling]   f({x_query}) = {stirling_val:.6f}")
                methods.append(("Стирлинг", lambda xx: stirling_interpolation(x, y, xx), "dashdot", "tab:red"))
            except Exception as e:
                print(f"Ошибка Стирлинга: {e}")
        if len(x) % 2 == 0:
            try:
                bessel_val = bessel_interpolation(x, y, x_query)
                print(f"[Bessel]     f({x_query}) = {bessel_val:.6f}")
                methods.append(("Бессель", lambda xx: bessel_interpolation(x, y, xx), "dotted", "tab:brown"))
            except Exception as e:
                print(f"Ошибка Бесселя: {e}")
    else:
        print("ВНИМАНИЕ: Узлы неравномерные — методы по конечным разностям не применяются!")

    # Визуализация всех методов
    plot_all_results(x, y, x_query, methods)

if __name__ == "__main__":
    main()
