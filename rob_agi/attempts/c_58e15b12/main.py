from rob_agi.colored_grid import ColoredGrid
from collections import deque
import math

def solve_58e15b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a lattice-like pattern from colored squares.
    
    The algorithm works as follows:
    1. Identifies all non-black squares in the input grid as source squares.
    2. Creates a distance map for each source square.
    3. Expands diagonally from source squares, creating a wave-like pattern.
    4. Determines colors based on distance to sources and grid edges.
    5. Applies a gradient effect and balances colors.
    6. Preserves original colored squares.
    
    Returns a new ColoredGrid with the transformed lattice-like pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    sources = []
    
    # Identify source squares
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                sources.append((r, c, input_grid.values[r][c]))
    
    # Create distance map
    distance_map = [[[math.inf for _ in range(len(sources))] for _ in range(cols)] for _ in range(rows)]
    for i, (sr, sc, _) in enumerate(sources):
        for r in range(rows):
            for c in range(cols):
                distance_map[r][c][i] = abs(r - sr) + abs(c - sc)
    
    # Wave expansion
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    visited = set(sources)
    queue = deque(sources)
    while queue:
        r, c, color = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc, color) not in visited:
                visited.add((nr, nc, color))
                queue.append((nr, nc, color))
    
    # Color determination
    for r in range(rows):
        for c in range(cols):
            if (r, c, 3) in visited and (r, c, 8) in visited:
                output_grid.values[r][c] = 6
            elif (r, c, 3) in visited:
                output_grid.values[r][c] = 3
            elif (r, c, 8) in visited:
                output_grid.values[r][c] = 8
    
    # Edge handling and gradient effect
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 0:
                edge_distance = min(r, c, rows-1-r, cols-1-c)
                if edge_distance < 3:
                    nearest_colors = [output_grid.values[nr][nc] for nr, nc in 
                                      [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                                      if 0 <= nr < rows and 0 <= nc < cols and output_grid.values[nr][nc] != 0]
                    if nearest_colors:
                        output_grid.values[r][c] = max(set(nearest_colors), key=nearest_colors.count)
    
    # Restore original squares
    for r, c, color in sources:
        output_grid.values[r][c] = color
    
    return output_grid
