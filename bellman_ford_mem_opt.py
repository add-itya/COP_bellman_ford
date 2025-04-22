from typing import List, Tuple
INF = float('inf')

def bellman_ford_mem_opt(n: int, edges: List[Tuple[int,int,int]], source: int) -> List[float]:
    prev = [INF]*n
    prev[source] = 0
    for _ in range(n-1):
        curr = prev.copy()
        for u, v, w in edges:
            if prev[u] != INF and prev[u] + w < curr[v]:
                curr[v] = prev[u] + w
        prev = curr
    return prev
