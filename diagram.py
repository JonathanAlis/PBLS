from graphviz import Digraph

dot = Digraph()

# Set direction: Left to Right
dot.attr(rankdir='LR')

# Add nodes (boxes)
dot.node('A', 'Input')
dot.node('B', 'Decomposition')
dot.node('C', 'Temporal Filtering')
dot.node('D', 'Phase Denoising')
dot.node('E', 'Reconstruction')
dot.node('F', 'Output')

# Add arrows (edges)
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')

# Export to PNG
dot.render('pipeline_diagram', format='png', cleanup=True)
