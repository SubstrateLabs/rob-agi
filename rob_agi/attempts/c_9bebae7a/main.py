from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9bebae7a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a symmetrical, expanded yellow shape.
    
    1. Analyze the input shape
    2. Determine primary expansion direction
    3. Apply shape-specific transformations
    4. Fill gaps and smooth edges
    5. Apply symmetry
    6. Extend patterns
    7. Adjust final size
    8. Center the shape
    9. Clean up and ensure symmetry
    10. Remove magenta
    """
    rows, cols = input_grid.get_dimensions()
    yellow_cells = set((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 4)
    
    if not yellow_cells:
        return input_grid  # No yellow cells, return original grid
    
    # Analyze the shape
    min_r, max_r = min(r for r, _ in yellow_cells), max(r for r, _ in yellow_cells)
    min_c, max_c = min(c for _, c in yellow_cells), max(c for _, c in yellow_cells)
    center_r, center_c = (min_r + max_r) // 2, (min_c + max_c) // 2
    
    # Determine expansion direction and apply transformations
    if center_r < rows // 2:
        yellow_cells.update((rows - 1 - r, c) for r, c in yellow_cells)  # Vertical mirror
    if center_c < cols // 2:
        yellow_cells.update((r, cols - 1 - c) for r, c in yellow_cells)  # Horizontal mirror
    
    # Fill gaps and smooth edges
    for _ in range(3):  # Repeat to create a more solid shape
        new_cells = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in yellow_cells:
                    neighbors = sum((r+dr, c+dc) in yellow_cells for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)])
                    if neighbors >= 2:
                        new_cells.add((r, c))
        yellow_cells.update(new_cells)
    
    # Apply symmetry
    symmetric_cells = set(yellow_cells)
    for r, c in yellow_cells:
        symmetric_cells.add((r, cols - 1 - c))
        symmetric_cells.add((rows - 1 - r, c))
    
    # Extend patterns and adjust size
    while len(symmetric_cells) / (rows * cols) < 0.25:  # If shape is too small
        new_cells = set()
        for r, c in symmetric_cells:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    new_cells.add((nr, nc))
        symmetric_cells.update(new_cells)
    
    # Center the shape
    r_offset = (rows - max(r for r, _ in symmetric_cells) - min(r for r, _ in symmetric_cells)) // 2
    c_offset = (cols - max(c for _, c in symmetric_cells) - min(c for _, c in symmetric_cells)) // 2
    centered_cells = {(r + r_offset, c + c_offset) for r, c in symmetric_cells}
    
    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c in centered_cells:
        if 0 <= r < rows and 0 <= c < cols:
            output_grid.values[r][c] = 4
    
    return output_grid
