from typing import List, Tuple
INF = float('inf')

def bellman_ford_time_opt(n: int, edges: List[Tuple[int,int,int]], source: int) -> List[float]:
    dist = [INF]*n
    dist[source] = 0
    for _ in range(n-1):
        updated = False
        for u, v, w in edges:
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    return dist
