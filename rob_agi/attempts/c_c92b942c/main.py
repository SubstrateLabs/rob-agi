from rob_agi.colored_grid import ColoredGrid

def solve_c92b942c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it 3x3, adding blue crosses around non-zero cells,
    and green corners to 3x3 blocks containing non-zero, non-blue, non-green cells.
    
    1. Create a 3x larger output grid
    2. Copy the input pattern to every 3x3 block in the output grid
    3. Add blue crosses around non-zero cells
    4. Add green corners to 3x3 blocks with non-zero, non-blue, non-green cells
    5. Return the transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    expanded_grid = ColoredGrid(values=[[0 for _ in range(3*cols)] for _ in range(3*rows)])
    
    # Copy the input pattern to every 3x3 block
    for i in range(rows):
        for j in range(cols):
            for di in range(3):
                for dj in range(3):
                    expanded_grid.set_cell(3*i+di, 3*j+dj, input_grid.get_cell(i, j))
    
    # Add blue crosses
    for r in range(3*rows):
        for c in range(3*cols):
            if expanded_grid.get_cell(r, c) not in [0, 1]:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 3*rows and 0 <= nc < 3*cols and expanded_grid.get_cell(nr, nc) == 0:
                        expanded_grid.set_cell(nr, nc, 1)
    
    # Add green corners
    for i in range(0, 3*rows, 3):
        for j in range(0, 3*cols, 3):
            if any(expanded_grid.get_cell(i+di, j+dj) not in [0, 1, 3] 
                   for di in range(3) for dj in range(3)):
                for corner_i, corner_j in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    if expanded_grid.get_cell(i+corner_i, j+corner_j) == 0:
                        expanded_grid.set_cell(i+corner_i, j+corner_j, 3)
    
    return expanded_grid
