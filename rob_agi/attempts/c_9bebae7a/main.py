from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9bebae7a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a symmetrical, expanded yellow shape.
    
    1. Analyze the input shape and determine its quadrant(s)
    2. Extend the shape towards the nearest corner(s)
    3. Apply appropriate symmetry (diagonal, edge, or both)
    4. Connect and refine the shape
    5. Adjust edges and maintain original characteristics
    6. Remove disconnected cells and perform final cleanup
    7. Create the final output grid
    """
    rows, cols = input_grid.get_dimensions()
    yellow_cells = set((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 4)
    
    if not yellow_cells:
        return input_grid  # No yellow cells, return original grid
    
    # Determine quadrant(s)
    r_center, c_center = rows // 2, cols // 2
    quadrants = set()
    for r, c in yellow_cells:
        if r < r_center:
            if c < c_center:
                quadrants.add(1)
            else:
                quadrants.add(2)
        else:
            if c < c_center:
                quadrants.add(3)
            else:
                quadrants.add(4)
    
    # Extend shape
    extended_cells = set(yellow_cells)
    for _ in range(max(rows, cols)):
        new_cells = set()
        for r, c in extended_cells:
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (1 in quadrants and nr <= r and nc <= c) or \
                       (2 in quadrants and nr <= r and nc >= c) or \
                       (3 in quadrants and nr >= r and nc <= c) or \
                       (4 in quadrants and nr >= r and nc >= c):
                        new_cells.add((nr, nc))
        if not new_cells:
            break
        extended_cells.update(new_cells)
    
    # Apply symmetry
    symmetric_cells = set(extended_cells)
    if len(quadrants) == 1:
        q = quadrants.pop()
        for r, c in extended_cells:
            if q == 1:
                symmetric_cells.add((rows-1-r, cols-1-c))
            elif q == 2:
                symmetric_cells.add((rows-1-r, c))
            elif q == 3:
                symmetric_cells.add((r, cols-1-c))
            else:
                symmetric_cells.add((rows-1-r, cols-1-c))
    else:
        for r, c in extended_cells:
            symmetric_cells.add((rows-1-r, c))
            symmetric_cells.add((r, cols-1-c))
            symmetric_cells.add((rows-1-r, cols-1-c))
    
    # Connect and refine
    for _ in range(2):
        new_cells = set()
        for r, c in symmetric_cells:
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors = sum((nr+dr, nc+dc) in symmetric_cells for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)])
                    if neighbors >= 2:
                        new_cells.add((nr, nc))
        symmetric_cells.update(new_cells)
    
    # Remove disconnected cells
    connected_cells = set()
    stack = [next(iter(symmetric_cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in symmetric_cells and (r, c) not in connected_cells:
            connected_cells.add((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in symmetric_cells:
                    stack.append((nr, nc))
    
    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    for r, c in connected_cells:
        output_grid.values[r][c] = 4
    
    return output_grid
