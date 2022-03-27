# generate_input.py

import os

def add_additional_combinations_dict(A, B, B_key):
    # Assume A is list of dicts (with the same keys) and B is a list of new
    # entries with key B_key.

    return [{**A[i // len(B)], **{B_key: B[i % len(B)]}} for i in range(len(A) * len(B))]

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str)
    parser.add_argument("--output_prefix", type=str, default="output")
    parser.add_argument("--ham_type", type=str, nargs='+')
    parser.add_argument("--rand_type", type=str, nargs='+')
    parser.add_argument("--L", type=int, nargs='+')
    parser.add_argument("--temp", type=float, nargs='+')
    parser.add_argument("--h_max", type=float, nargs='+', default=[1])
    parser.add_argument("--J_max", type=float, nargs='+', default=[1])
    parser.add_argument("--delta", type=str, nargs='+')
    parser.add_argument("--no_of_processes", type=int, nargs='+')
    parser.add_argument("--no_of_samples", type=int, nargs='+')

    args = parser.parse_args()

    params = {"ham_type": args.ham_type, "rand_type": args.rand_type,
              "L": args.L, "temp": args.temp, "h_max": args.h_max, "J_max": args.J_max, 
              "delta": args.delta, "no_of_processes": args.no_of_processes, 
              "no_of_samples": args.no_of_samples}

    comb = [{"L": args.L[i]} for i in range(len(args.L))]
    for key in params.keys():
        if key != "L":
            comb = add_additional_combinations_dict(comb, params[key], key)

    if args.machine == "Feynman":
        path = './run/input/feynman/'
    elif args.machine == "Feynman2":
        path = './run/input/feynman2/'
    elif args.machine == "Icarus":
        path = './run/input/icarus/'
    else:
        raise ValueError

    for i, comb_i in enumerate(comb):  
        fh = open(path + 'input' + str(i) + '.txt', 'w')
        fh.write('#' + args.machine + '\n')
        fh.write('\n')
        fh.write("output_prefix= " + args.output_prefix)
        fh.write('\n')
        for key in comb_i.keys():
            fh.write(key + '= ' + str(comb_i[key]) + '\n')
        fh.write('END')
        fh.close()