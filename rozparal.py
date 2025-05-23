import math
import random
import time
from sympy import primerange
from concurrent.futures import ThreadPoolExecutor, as_completed


def base_base(n, c=3.38):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    B = c * math.exp(0.5 * math.sqrt(ln_n * ln_ln_n))
    return list(primerange(2, int(B)))


def is_smooth(value, base):
    exp_vector = [0] * len(base)
    for i, p in enumerate(base):
        while value % p == 0:
            value //= p
            exp_vector[i] += 1
    return exp_vector if value == 1 else None


def generate_relation(alpha, mod, base):
    while True:
        k = random.randint(1, mod - 1)
        ak = pow(alpha, k, mod)
        exp_vector = is_smooth(ak, base)
        if exp_vector:
            return exp_vector, k


def generate_linear_system_parallel(alpha, mod, base, num_eq, workers=4):
    equations, ks = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_relation, alpha, mod, base) for _ in range(num_eq * 2)]
        for future in as_completed(futures):
            exp_vector, k = future.result()
            equations.append(exp_vector)
            ks.append(k)
            if len(equations) >= num_eq:
                break
    return equations, ks


def gcd(a, b):
    u0, u1 = 1, 0
    v0, v1 = 0, 1
    while b != 0:
        q, a, b = a // b, b, a % b
        u0, u1 = u1, u0 - q * u1
        v0, v1 = v1, v0 - q * v1
    return a, u0, v0


def mod_inverse(a, mod):
    d, u, _ = gcd(a, mod)
    return u % mod if d == 1 else None


def solve_linear_mod_system(A, b, mod):
    A = [row[:] for row in A]
    b = b[:]
    n, m = len(A), len(A[0])

    for i in range(min(n, m)):
        pivot = next((r for r in range(i, n) if A[r][i] % mod != 0), None)
        if pivot is None:
            continue
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

        inv = mod_inverse(A[i][i], mod)
        if inv is None:
            continue

        A[i] = [(val * inv) % mod for val in A[i]]
        b[i] = (b[i] * inv) % mod

        for r in range(n):
            if r != i:
                factor = A[r][i]
                A[r] = [(A[r][c] - factor * A[i][c]) % mod for c in range(m)]
                b[r] = (b[r] - factor * b[i]) % mod

    return [b[i] % mod for i in range(m)]


def try_log_beta(alpha, beta, mod, base, log_base):
    while True:
        k = random.randint(0, mod - 1)
        val = (beta * pow(alpha, k, mod)) % mod
        di = is_smooth(val, base)
        if di:
            log_beta_val = (sum(d * lp for d, lp in zip(di, log_base)) - k) % (mod - 1)
            return log_beta_val, k, di, val


def index_calculus(alpha, beta, n, max_time=300, threads=4):
    mod = n
    base = base_base(n)
    t = len(base)

    print("Факторна база S =", base)

    start_time = time.time()
    attempt = 0

    while time.time() - start_time < max_time:
        attempt += 1
        print(f"\nСпроба №{attempt} — генерація системи...")

        equations, ks = generate_linear_system_parallel(alpha, mod, base, t + 5, workers=threads)

        solution = solve_linear_mod_system(equations, ks, n - 1)
        if not solution:
            print("Не вдалося розв’язати систему.")
            continue

        print("logα(p_i):")
        for pi, log_val in zip(base, solution):
            print(f"logα({pi}) = {log_val} mod {n - 1}")

        print("Обчислення logα(beta)...")
        log_beta_val, _, _, _ = try_log_beta(alpha, beta, mod, base, solution)

        if pow(alpha, log_beta_val, mod) == beta:
            total_time = time.time() - start_time
            print(f"Відповідь: logα({beta}) = {log_beta_val} (mod {n - 1})")
            print(f"Час виконання: {total_time:.2f} секунд")
            return log_beta_val

        print("Перевірка не пройдена, спроба ще раз...")

    total_time = time.time() - start_time
    print("Час вичерпано. Відповідь не знайдена.")
    print(f"Загальний час: {total_time:.2f} секунд")
    return None


if __name__ == "__main__":
    alpha = 201
    beta = 627
    n = 733
    index_calculus(alpha, beta, n, max_time=300, threads=4)
