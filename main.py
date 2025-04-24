"""
has 3 predefined graphs, measures runtime, peak memory, and prints results
"""
import time, tracemalloc, sys, os

sys.path.append(os.path.dirname(__file__))

import bellman_ford_basic as bf_basic
import bellman_ford_time_opt as bf_time
import bellman_ford_mem_opt as bf_mem

TestGraph = tuple[int, list[tuple[int,int,int]], int]

graphs: list[TestGraph] = [
    (5, [
        (0, 1, 5),
        (1, 2, 1),
        (1, 3, 2),
        (2, 4, 1),
        (4, 3, -1)
    ], 0),
    (4, [
        (0, 1, 4),
        (1, 2, -2),
        (2, 3, 5),
        (3, 1, -1)
    ], 0),(3, [
        (0, 1, 2),
        (1, 2, 4),
        (0, 2, 6)    
    ], 0),

    (6, [
        (0, 1, 7), (1, 0, 7),
        (0, 2, 9), (2, 0, 9),
        (0, 5,14), (5, 0,14),
        (1, 2,10), (2, 1,10),
        (1, 3,15), (3, 1,15),
        (2, 3,11), (3, 2,11),
        (2, 5, 2), (5, 2, 2),
        (3, 4, 6), (4, 3, 6),
        (4, 5, 9), (5, 4, 9)
    ], 0),

    (7, [
        (0, 1, 3),
        (0, 2, 2),
        (2, 1, 1),
        (1, 3, 4),
        (2, 3, 8),
        (3, 4, 2),
        (4, 5, 3),
        (5, 6, 1),
        (4, 6, 6)
    ], 0)
]

algos = [
    ("Basic DP", bf_basic.bellman_ford_basic),
    ("Early‑exit", bf_time.bellman_ford_time_opt),
    ("Memory‑opt", bf_mem.bellman_ford_mem_opt)
]

def run_algo(name: str, func, n, edges, src):
    tracemalloc.start()
    t0 = time.perf_counter()
    dist = func(n, edges, src)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024, dist

def main():
    print("\n===== Bellman–Ford shoot‑out =====\n")
    for g_idx, (n, edges, src) in enumerate(graphs, start=1):
        print(f"Graph {g_idx}: vertices={n}, edges={len(edges)}, source={src}")
        for name, func in algos:
            elapsed, mem_kib, dist = run_algo(name, func, n, edges, src)
            print(f"  {name:<12}  time: {elapsed*1e3:8.3f} ms   peak mem: {mem_kib:8.1f} KiB   dists: {dist}")
        print()
    print("Done.")

if __name__ == "__main__":
    main()
