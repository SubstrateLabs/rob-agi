from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9bebae7a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a symmetrical, expanded yellow shape.
    
    1. Analyze the input shape and determine its position
    2. Create a symmetrical version based on the shape's position
    3. Expand the shape towards edges while maintaining symmetry
    4. Refine the shape to ensure connectivity and maintain original characteristics
    5. Create the final output grid
    """
    rows, cols = input_grid.get_dimensions()
    yellow_cells = set((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 4)
    
    if not yellow_cells:
        return input_grid  # No yellow cells, return original grid
    
    # Determine shape position
    min_r = min(r for r, _ in yellow_cells)
    max_r = max(r for r, _ in yellow_cells)
    min_c = min(c for _, c in yellow_cells)
    max_c = max(c for _, c in yellow_cells)
    
    # Create symmetrical version
    symmetric_cells = set(yellow_cells)
    if min_r < rows - max_r - 1:  # Shape is closer to top
        for r, c in yellow_cells:
            symmetric_cells.add((rows - 1 - r, c))
    elif min_r > rows - max_r - 1:  # Shape is closer to bottom
        for r, c in yellow_cells:
            symmetric_cells.add((rows - 1 - r, c))
    if min_c < cols - max_c - 1:  # Shape is closer to left
        for r, c in yellow_cells:
            symmetric_cells.add((r, cols - 1 - c))
    elif min_c > cols - max_c - 1:  # Shape is closer to right
        for r, c in yellow_cells:
            symmetric_cells.add((r, cols - 1 - c))
    
    # Expand shape
    expanded_cells = set(symmetric_cells)
    while True:
        new_cells = set()
        for r, c in expanded_cells:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    new_cells.add((nr, nc))
        if not new_cells - expanded_cells:
            break
        expanded_cells.update(new_cells)
    
    # Refine shape
    refined_cells = set()
    for r, c in expanded_cells:
        neighbors = sum((r+dr, c+dc) in expanded_cells for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)])
        if neighbors >= 2 or (r, c) in symmetric_cells:
            refined_cells.add((r, c))
    
    # Ensure connectivity
    connected_cells = set()
    stack = [next(iter(refined_cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in refined_cells and (r, c) not in connected_cells:
            connected_cells.add((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in refined_cells:
                    stack.append((nr, nc))
    
    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c in connected_cells:
        output_grid.values[r][c] = 4
    
    return output_grid
