from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9bebae7a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a symmetrical, expanded yellow shape.
    
    1. Analyze the input shape
    2. Apply vertical and horizontal symmetry
    3. Expand the shape to fill gaps and create a more solid form
    4. Ensure perfect symmetry
    5. Balance the shape within the grid
    6. Remove any disconnected yellow cells
    7. Create the final output grid
    """
    rows, cols = input_grid.get_dimensions()
    yellow_cells = set((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 4)
    
    if not yellow_cells:
        return input_grid  # No yellow cells, return original grid
    
    # Apply symmetry
    symmetric_cells = set(yellow_cells)
    for r, c in yellow_cells:
        symmetric_cells.add((rows - 1 - r, c))  # Vertical symmetry
        symmetric_cells.add((r, cols - 1 - c))  # Horizontal symmetry
        symmetric_cells.add((rows - 1 - r, cols - 1 - c))  # Both
    
    # Expand shape and fill gaps
    for _ in range(3):
        new_cells = set()
        for r, c in symmetric_cells:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors = sum((nr+dr, nc+dc) in symmetric_cells for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)])
                    if neighbors >= 2:
                        new_cells.add((nr, nc))
        symmetric_cells.update(new_cells)
    
    # Ensure perfect symmetry
    perfect_symmetric_cells = set()
    for r, c in symmetric_cells:
        perfect_symmetric_cells.add((r, c))
        perfect_symmetric_cells.add((rows - 1 - r, c))
        perfect_symmetric_cells.add((r, cols - 1 - c))
        perfect_symmetric_cells.add((rows - 1 - r, cols - 1 - c))
    
    # Balance the shape
    r_min, r_max = min(r for r, _ in perfect_symmetric_cells), max(r for r, _ in perfect_symmetric_cells)
    c_min, c_max = min(c for _, c in perfect_symmetric_cells), max(c for _, c in perfect_symmetric_cells)
    r_offset = (rows - r_max - r_min) // 2
    c_offset = (cols - c_max - c_min) // 2
    balanced_cells = {(r + r_offset, c + c_offset) for r, c in perfect_symmetric_cells}
    
    # Remove disconnected cells
    connected_cells = set()
    stack = [next(iter(balanced_cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in balanced_cells and (r, c) not in connected_cells:
            connected_cells.add((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in balanced_cells:
                    stack.append((nr, nc))
    
    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c in connected_cells:
        if 0 <= r < rows and 0 <= c < cols:
            output_grid.values[r][c] = 4
    
    return output_grid
