import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Define the architecture from your description
architecture = {
    0: [(0, 'avg_pool'), (0, 'sep_conv_7')],
    1: [(1, 'max_pool'), (2, 'conv_3')],
    2: [(1, 'max_pool'), (2, 'sep_conv_5')],
    3: [(1, 'dil_conv_3'), (3, 'avg_pool')],
    4: [(3, 'dil_conv_3'), (1, 'avg_pool')],
    5: [(3, 'sep_conv_3'), (0, 'conv_3')],
    6: [(7, 'dil_conv_5'), (7, 'sep_conv_7')],
    7: [(5, 'conv_5'), (2, 'identity')],
    8: [(6, 'avg_pool'), (3, 'sep_conv_3')],
    9: [(10, 'sep_conv_7'), (10, 'max_pool')]  # Note: Node 10 doesn't exist, might be input node
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes
for node in range(10):
    G.add_node(node, label=f"Node {node}")

# Add edges with operations
for target_node, edges in architecture.items():
    for source_node, operation in edges:
        if source_node <= 9:  # Only add if source node exists
            G.add_edge(source_node, target_node, operation=operation)

# Create visualization
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=2)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=2000, alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Draw edges with operation labels
edge_labels = nx.get_edge_attributes(G, 'operation')
nx.draw_networkx_edges(G, pos, edge_color='gray', 
                      arrows=True, arrowsize=20, width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                           font_size=8, font_color='darkred')

plt.title("NAS Cell Architecture Visualization", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# Alternative: Create a more organized hierarchical layout
plt.figure(figsize=(15, 10))

# Create a layered layout (nodes in rows)
layer_pos = {}
for i in range(10):
    layer_pos[i] = (i, 0)  # All nodes in one row

# Reorganize for better readability
layer_pos = {
    0: (0, 0),
    1: (1, 1),
    2: (2, 0),
    3: (3, 1),
    4: (4, 0),
    5: (5, 1),
    6: (6, 0),
    7: (7, 1),
    8: (8, 0),
    9: (9, 1)
}

# Draw the graph with hierarchical layout
nx.draw_networkx_nodes(G, layer_pos, node_color='lightblue', 
                      node_size=2000, alpha=0.8)
nx.draw_networkx_labels(G, layer_pos, font_size=10, font_weight='bold')

# Draw curved edges for better visualization
for edge in G.edges():
    source, target = edge
    rad = 0.2 if (source + target) % 2 == 0 else -0.2
    nx.draw_networkx_edges(G, layer_pos, edgelist=[edge], 
                          connectionstyle=f'arc3,rad={rad}',
                          arrows=True, arrowsize=20, width=2)

# Add edge labels with operations
edge_labels = nx.get_edge_attributes(G, 'operation')
nx.draw_networkx_edge_labels(G, layer_pos, edge_labels=edge_labels,
                           font_size=8, font_color='darkred', 
                           bbox=dict(alpha=0))

plt.title("Hierarchical View of NAS Cell Architecture", 
          fontsize=16, fontweight='bold')
plt.xlim(-1, 10)
plt.ylim(-1, 2)
plt.axis('off')
plt.tight_layout()
plt.show()

# Print the architecture in a structured way
print("="*60)
print("ARCHITECTURE SUMMARY")
print("="*60)
for node in range(10):
    inputs = architecture[node]
    print(f"\nNode {node}:")
    for i, (source, op) in enumerate(inputs):
        print(f"  Input {i}: from Node {source} via {op}")
print("="*60)

# Additional: Create adjacency matrix view
print("\nAdjacency Matrix (with operations):")
print("-"*40)
adj_matrix = np.zeros((10, 10), dtype=object)

for target, edges in architecture.items():
    for source, op in edges:
        if source <= 9:
            adj_matrix[source, target] = op

print("     ", end="")
for j in range(10):
    print(f"{j:^8}", end="")
print()

for i in range(10):
    print(f"{i:2} | ", end="")
    for j in range(10):
        op = adj_matrix[i, j]
        if op:
            print(f"{op[:6]:^8}", end="")
        else:
            print(f"{'':^8}", end="")
    print()