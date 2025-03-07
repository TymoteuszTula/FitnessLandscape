from NewRunFunction import single_run

if __name__ == "__main__":
    Ls = [4, 6, 8, 10]
    temps = [0.0, 1.0]
    samples = 1000
    processes = 24

    for L in Ls:
        for temp in temps:
            single_run(L, temp, samples, processes)