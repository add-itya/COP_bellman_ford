from typing import List, Tuple

INF = float('inf')

def bellman_ford_basic(n: int, edges: List[Tuple[int,int,int]], source: int) -> List[float]:
    """
    returns list of distances from src to every node. Assumes no negative cycles
    n: number of vertices ranging from 0 to n-1
    edges: list of (u, v, w) directed edges
    """
    # table[i][v] = shortest distance with ≤ i edges
    table = [[INF]*n for _ in range(n)]
    table[0][source] = 0

    for i in range(1, n):
        table[i] = table[i-1].copy()
        # relaxing every edge, reading from previous row
        for u, v, w in edges:
            if table[i-1][u] != INF and table[i-1][u] + w < table[i][v]:
                table[i][v] = table[i-1][u] + w
    return table[-1]
