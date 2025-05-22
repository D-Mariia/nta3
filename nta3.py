import math
import random
import time
from sympy import primerange


def base_base(n, c=3.38):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    B = c * math.exp(0.5 * math.sqrt(ln_n * ln_ln_n))
    p_i = list(primerange(2, int(B)))
    return p_i


def is_smooth(value, base):
    exp_vector = [0] * len(base)
    for i, p in enumerate(base):
        while value % p == 0:
            value //= p
            exp_vector[i] += 1
    if value == 1:
        return exp_vector
    return None


def generate_linear_system(alpha, mod, p_i, num_eq):
    equations = []
    ks = []
    while len(equations) < num_eq:
        k = random.randint(1, mod - 1)
        ak = pow(alpha, k, mod)
        exp_vector = is_smooth(ak, p_i)
        if exp_vector is not None:
            equations.append(exp_vector)
            ks.append(k)
    return equations, ks


def gcd(a, b):
    u0, u1 = 1, 0
    v0, v1 = 0, 1

    r0, r1 = a, b

    while r1 != 0:
        q = r0 // r1
        r2 = r0 % r1

        r0, r1 = r1, r2

        u0, u1 = u1, u0 - q * u1
        v0, v1 = v1, v0 - q * v1

    return r0, u0, v0


def mod_inverse(a):
    d, u, _ = gcd(a, n)
    if d != 1:
        return None
    return u % n


def solve_linear_mod_system(A, b, mod):
    A = [row[:] for row in A]
    b = b[:]
    n = len(A)
    m = len(A[0])

    for i in range(min(n, m)):
        pivot = None
        for r in range(i, n):
            if A[r][i] % mod != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

        inv = mod_inverse(A[i][i])
        if inv is None:
            continue

        for col in range(i, m):
            A[i][col] = (A[i][col] * inv) % mod
        b[i] = (b[i] * inv) % mod

        for r in range(n):
            if r != i:
                factor = A[r][i]
                for col in range(i, m):
                    A[r][col] = (A[r][col] - factor * A[i][col]) % mod
                b[r] = (b[r] - factor * b[i]) % mod

    solution = [0] * m
    for i in range(m):
        solution[i] = b[i] % mod
    return solution


def log_beta(alpha, beta, mod, p_i, log_p_i):
    n = mod
    while True:
        k = random.randint(0, n - 1)
        val = (beta * pow(alpha, k, n)) % n
        di = is_smooth(val, p_i)
        if di is not None:
            log_beta_val = sum(d * lp for d, lp in zip(di, log_p_i)) - k
            return log_beta_val % (n - 1), k, di, val

alpha = 201
beta = 627
n = 733
mod = n
p_i = base_base(n)
t = len(p_i)

print("Факторна база S =", p_i)

MAX_TIME = 300

start_time = time.time()

attempt = 0
while True:
    current_time = time.time()
    if current_time - start_time > MAX_TIME:
        print("Час роботи вичерпано (5 хвилин). Програма завершується.")
        break

    attempt += 1
    attempt_start = time.time()

    equations, ks = generate_linear_system(alpha, mod, p_i, t + 5)

    print(f"\nСпроба №{attempt}")
    print("Система лінійних рівнянь:")
    for i in range(len(equations)):
        lhs = f"{ks[i]}"
        rhs = " + ".join(f"{eq}*logα({p})" for eq, p in zip(equations[i], p_i) if eq > 0)
        print(f"{lhs} = {rhs} mod {n - 1}")

    solution = solve_linear_mod_system(equations, ks, n - 1)

    print("\n(logα(p_i) mod", n - 1, "):")
    if solution:
        for pi, log_val in zip(p_i, solution):
            print(f"logα({pi}) = {log_val} mod {n - 1}")
    else:
        print("Розв’язок не знайдено")
        continue

    log_beta_val, l, di, val = log_beta(alpha, beta, mod, p_i, solution)

    print(f"\nСпроба: logα({beta}) = {log_beta_val} (mod {mod - 1})")

    attempt_end = time.time()
    print(f"Час цієї спроби: {attempt_end - attempt_start:.4f} секунд")

    if pow(alpha, log_beta_val, mod) == beta:
        print(f"\nВідповідь підтверджена: logα({beta}) = {log_beta_val} (mod {mod - 1})")
        break
    else:
        print("Перевірка не пройдена, запускаємо ще ...\n")

end_time = time.time()
print(f"\nЗагальний час роботи: {end_time - start_time:.4f} секунд")
