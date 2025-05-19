import math
import random
from sympy import primerange
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed


def base_base(n, c=3.38):
    ln_n = math.log(n)
    ln_ln_n = math.log(ln_n)
    B = c * math.exp(0.5 * math.sqrt(ln_n * ln_ln_n))
    p_i = list(primerange(2, B))
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


def generate_one_equation(alpha, mod, p_i):
    while True:
        k = random.randint(1, mod - 1)
        ak = pow(alpha, k, mod)
        exp_vector = is_smooth(ak, p_i)
        if exp_vector is not None:
            return exp_vector, k


def generate_linear_system_parallel(alpha, mod, p_i, num_eq):
    equations = []
    ks = []
    with ProcessPoolExecutor() as executor:
        while len(equations) < num_eq:
            futures = [executor.submit(generate_one_equation, alpha, mod, p_i) for _ in range(num_eq - len(equations))]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    exp_vec, k = result
                    equations.append(exp_vec)
                    ks.append(k)
    return equations, ks


def brute_force_solve(matrix, vector, mod):
    n = len(matrix[0])
    for candidate in product(range(mod), repeat=n):
        ok = True
        for row, b in zip(matrix, vector):
            if sum((a * x for a, x in zip(row, candidate))) % mod != b % mod:
                ok = False
                break
        if ok:
            return list(candidate)
    return None


def try_log_beta(alpha, beta, mod, p_i, log_p_i):
    n = mod
    while True:
        k = random.randint(0, n - 1)
        val = (beta * pow(alpha, k, n)) % n
        di = is_smooth(val, p_i)
        if di is not None:
            log_beta = sum(d * lp for d, lp in zip(di, log_p_i)) - k
            return log_beta % (n - 1), k, di, val


def log_beta_parallel(alpha, beta, mod, p_i, log_p_i):
    with ProcessPoolExecutor() as executor:
        while True:
            futures = [executor.submit(try_log_beta, alpha, beta, mod, p_i, log_p_i) for _ in range(10)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    return result


if __name__ == "__main__":
    alpha = 10
    beta = 17
    n = 47
    mod = n
    p_i = base_base(n)
    t = len(p_i)

    print("Факторна база S =", p_i)

    equations, ks = generate_linear_system_parallel(alpha, mod, p_i, t + 5)

    print("\nСистема лінійних рівнянь:")
    for i in range(len(equations)):
        lhs = f"{ks[i]}"
        rhs = " + ".join(f"{eq}*logα({p})" for eq, p in zip(equations[i], p_i) if eq > 0)
        print(f"{lhs} = {rhs} mod {n - 1}")

    solution = brute_force_solve(equations, ks, n - 1)

    print("\n(logα(p_i) mod", n - 1, "):")
    if solution:
        for pi, log_val in zip(p_i, solution):
            print(f"logα({pi}) = {log_val} mod {n - 1}")
    else:
        print("Розв’язок не знайдено")
        exit(1)

    log_b, l, di, val = log_beta_parallel(alpha, beta, mod, p_i, solution)

    print(f"\nВідповідь : logα({beta}) = {log_b} (mod {mod - 1})")