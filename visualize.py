import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import time
import tracemalloc
from typing import List, Tuple, Callable
import numpy as np

import bellman_ford_basic as bf_basic
import bellman_ford_time_opt as bf_time
import bellman_ford_mem_opt as bf_mem
from main import graphs, algos

def create_graph(n: int, edges: List[Tuple[int, int, int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G

def draw_graph(G: nx.DiGraph, distances: List[float] = None, active_edges: List[Tuple[int, int]] = None, active_nodes: List[int] = None, pos=None):
    fig = plt.figure(figsize=(10, 6))
    plt.clf()
    if pos is None:
        pos = nx.spring_layout(G, k=2, iterations=50)  # k=2 for more spread out layout
    
    # Draw edges with different colors for active edges
    edge_colors = []
    edge_widths = []
    for (u, v) in G.edges():
        if active_edges and (u, v) in active_edges:
            edge_colors.append('red')
            edge_widths.append(2.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    
    # Draw nodes with colors based on state
    node_colors = []
    node_sizes = []
    for i in G.nodes():
        if active_nodes and i in active_nodes:
            if distances and distances[i] == 0:  # Source node
                node_colors.append('lightgreen')
            else:  # Currently being processed
                node_colors.append('orange')
            node_sizes.append(600)  # Larger size for active nodes
        else:
            if distances and distances[i] == float('inf'):
                node_colors.append('lightblue')
            elif distances and distances[i] == 0:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightyellow')
            node_sizes.append(500)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    
    # Draw node labels
    labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    plt.axis('off')
    return fig, pos

def step_through_basic(n: int, edges: List[Tuple[int, int, int]], source: int):
    INF = float('inf')
    table = [[INF]*n for _ in range(n)]
    table[0][source] = 0
    
    for i in range(1, n):
        table[i] = table[i-1].copy()
        active_nodes = set()
        active_edges = []
        
        for u, v, w in edges:
            active_nodes.add(u)
            active_nodes.add(v)
            active_edges.append((u, v))
            
            if table[i-1][u] != INF and table[i-1][u] + w < table[i][v]:
                table[i][v] = table[i-1][u] + w
            
            yield table[i], list(active_edges), list(active_nodes)
            # Clear active states for next iteration
            active_nodes = set()
            active_edges = []

def step_through_time_opt(n: int, edges: List[Tuple[int, int, int]], source: int):
    INF = float('inf')
    dist = [INF]*n
    dist[source] = 0
    
    for _ in range(n-1):
        updated = False
        active_nodes = set()
        active_edges = []
        
        for u, v, w in edges:
            active_nodes.add(u)
            active_nodes.add(v)
            active_edges.append((u, v))
            
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
            
            yield dist.copy(), list(active_edges), list(active_nodes)
            # Clear active states for next iteration
            active_nodes = set()
            active_edges = []
            
        if not updated:
            break

def step_through_mem_opt(n: int, edges: List[Tuple[int, int, int]], source: int):
    INF = float('inf')
    prev = [INF]*n
    prev[source] = 0
    
    for _ in range(n-1):
        curr = prev.copy()
        active_nodes = set()
        active_edges = []
        
        for u, v, w in edges:
            active_nodes.add(u)
            active_nodes.add(v)
            active_edges.append((u, v))
            
            if prev[u] != INF and prev[u] + w < curr[v]:
                curr[v] = prev[u] + w
            
            yield curr.copy(), list(active_edges), list(active_nodes)
            # Clear active states for next iteration
            active_nodes = set()
            active_edges = []
        
        prev = curr

def run_comparison():
    results = []
    for g_idx, (n, edges, src) in enumerate(graphs, start=1):
        graph_results = []
        for name, func in algos:
            tracemalloc.start()
            t0 = time.perf_counter()
            dist = func(n, edges, src)
            elapsed = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            graph_results.append({
                'name': name,
                'time_ms': elapsed * 1e3,
                'memory_kib': peak / 1024,
                'distances': dist
            })
        results.append(graph_results)
    return results

def main():
    st.title("Bellman-Ford Algorithm Visualization")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_graph = st.sidebar.selectbox(
        "Select Test Graph",
        options=range(len(graphs)),
        format_func=lambda x: f"Graph {x+1} ({graphs[x][0]} vertices, {len(graphs[x][1])} edges)"
    )
    
    selected_algo = st.sidebar.selectbox(
        "Select Algorithm",
        options=["Basic DP", "Early-exit", "Memory-opt"]
    )
    
    # Get the selected graph
    n, edges, source = graphs[selected_graph]
    G = create_graph(n, edges)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Step Through Algorithm", "Compare All"])
    
    with tab1:
        st.subheader("Graph Visualization")
        
        # Initialize or reset state when graph or algorithm changes
        if ('selected_graph' not in st.session_state or 
            'selected_algo' not in st.session_state or
            st.session_state.selected_graph != selected_graph or 
            st.session_state.selected_algo != selected_algo):
            
            st.session_state.selected_graph = selected_graph
            st.session_state.selected_algo = selected_algo
            st.session_state.step_idx = 0
            st.session_state.graph_pos = None
            
            # Get stepper function based on selected algorithm
            if selected_algo == "Basic DP":
                stepper = step_through_basic
            elif selected_algo == "Early-exit":
                stepper = step_through_time_opt
            else:
                stepper = step_through_mem_opt
            
            st.session_state.all_steps = list(stepper(n, edges, source))
        
        # Navigation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⏮️ Reset"):
                st.session_state.step_idx = 0
        with col2:
            if st.button("⏪ Previous Step") and st.session_state.step_idx > 0:
                st.session_state.step_idx -= 1
        with col3:
            if st.button("Next Step ⏩") and st.session_state.step_idx < len(st.session_state.all_steps) - 1:
                st.session_state.step_idx += 1
        
        # Display current step
        current_step = st.session_state.all_steps[st.session_state.step_idx]
        current_distances = current_step[0]  # Unpack the step data
        active_edges = current_step[1] if len(current_step) > 1 else []
        active_nodes = current_step[2] if len(current_step) > 2 else []
        
        fig, pos = draw_graph(G, current_distances, active_edges, active_nodes, st.session_state.graph_pos)
        if st.session_state.graph_pos is None:
            st.session_state.graph_pos = pos
        st.pyplot(fig)
        
        # Display step information
        st.write(f"Step {st.session_state.step_idx + 1}/{len(st.session_state.all_steps)}")
        
        # Show active elements
        if active_edges or active_nodes:
            st.write("**Currently Processing:**")
            if active_edges:
                edge_str = ", ".join(f"({u}→{v})" for u, v in active_edges)
                st.write(f"Examining edge(s): {edge_str}")
            if active_nodes:
                node_str = ", ".join(str(n) for n in active_nodes)
                st.write(f"Active nodes: {node_str}")
        
        st.write("Current distances from source:")
        distances_dict = {f"Node {i}": "∞" if d == float('inf') else f"{d:.1f}" 
                         for i, d in enumerate(current_distances)}
        st.json(distances_dict)
    
    with tab2:
        st.subheader("Algorithm Comparison")
        if st.button("Run Comparison"):
            results = run_comparison()
            
            for g_idx, graph_results in enumerate(results):
                st.write(f"\n### Graph {g_idx + 1}")
                st.write(f"Vertices: {graphs[g_idx][0]}, Edges: {len(graphs[g_idx][1])}")
                
                # Create performance comparison table
                perf_data = []
                for result in graph_results:
                    perf_data.append({
                        'Algorithm': result['name'],
                        'Time': f"{result['time_ms']:.3f} ms",
                        'Peak Memory': f"{result['memory_kib']:.1f} KiB"
                    })
                st.table(perf_data)
                
                # Display distances in a more readable format
                st.write("Final distances from source:")
                for result in graph_results:
                    st.write(f"**{result['name']}:**")
                    cols = st.columns(4)  # Display in 4 columns for better layout
                    distances = result['distances']
                    col_size = (len(distances) + 3) // 4  # Distribute nodes across columns
                    
                    for col_idx, col in enumerate(cols):
                        start_idx = col_idx * col_size
                        end_idx = min(start_idx + col_size, len(distances))
                        if start_idx < len(distances):
                            with col:
                                for i in range(start_idx, end_idx):
                                    d = distances[i]
                                    value = "∞" if d == float('inf') else f"{d:.1f}"
                                    st.write(f"Node {i}: {value}")
                st.write("---")  # Add separator between graphs

if __name__ == "__main__":
    main() 