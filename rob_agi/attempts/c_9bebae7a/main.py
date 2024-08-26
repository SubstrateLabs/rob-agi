from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9bebae7a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a symmetrical, expanded yellow shape.
    
    1. Identify the yellow (4) shape
    2. Create a horizontal mirror image
    3. Expand vertically
    4. Fill horizontally
    5. Refine the shape (fill gaps, smooth edges)
    6. Center the shape
    7. Ensure perfect symmetry
    """
    rows, cols = input_grid.get_dimensions()
    yellow_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 4]
    
    if not yellow_cells:
        return input_grid  # No yellow cells, return original grid
    
    # Find boundaries of the yellow shape
    min_r = min(r for r, _ in yellow_cells)
    max_r = max(r for r, _ in yellow_cells)
    min_c = min(c for _, c in yellow_cells)
    max_c = max(c for _, c in yellow_cells)
    
    # Create horizontal mirror image
    mirrored_cells = set(yellow_cells + [(r, cols - 1 - c) for r, c in yellow_cells])
    
    # Vertical expansion
    height = max_r - min_r + 1
    new_height = min(rows, min_r + 2 * height)
    expanded_cells = set(mirrored_cells.union((r + height, c) for r, c in mirrored_cells if r + height < new_height))
    
    # Horizontal filling
    center_c = cols // 2
    target_width = int(cols * 0.9)  # Fill about 90% of the width
    while max(c for _, c in expanded_cells) - min(c for _, c in expanded_cells) < target_width:
        new_cells = set()
        for r, c in expanded_cells:
            for dc in [-1, 1]:
                new_c = c + dc
                if 0 <= new_c < cols:
                    new_cells.add((r, new_c))
        expanded_cells.update(new_cells)
    
    # Shape refinement
    refined_cells = set(expanded_cells)
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in refined_cells:
                neighbors = sum((r+dr, c+dc) in refined_cells for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)])
                if neighbors >= 3:
                    refined_cells.add((r, c))
    
    # Ensure symmetry
    symmetric_cells = set(refined_cells)
    for r, c in refined_cells:
        symmetric_cells.add((r, cols - 1 - c))
    
    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c in symmetric_cells:
        output_grid.values[r][c] = 4
    
    return output_grid
