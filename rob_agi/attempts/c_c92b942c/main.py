from rob_agi.colored_grid import ColoredGrid

def solve_c92b942c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it 3x3, adding blue crosses around non-zero cells,
    and green corners to 3x3 blocks containing non-zero, non-blue, non-green cells.
    
    1. Create a 3x larger output grid
    2. Copy and expand the input pattern
    3. Add blue crosses around non-zero cells
    4. Add green corners to 3x3 blocks with non-zero, non-blue, non-green cells
    5. Return the transformed grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(3*cols)] for _ in range(3*rows)])
    
    # Copy and expand the input pattern
    for i in range(rows):
        for j in range(cols):
            if input_grid.get_cell(i, j) != 0:
                for di in range(3):
                    for dj in range(3):
                        output_grid.set_cell(3*i+di, 3*j+dj, input_grid.get_cell(i, j))
    
    # Add blue crosses
    for i in range(3*rows):
        for j in range(3*cols):
            if output_grid.get_cell(i, j) not in [0, 1]:
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 3*rows and 0 <= nj < 3*cols and output_grid.get_cell(ni, nj) == 0:
                        output_grid.set_cell(ni, nj, 1)
    
    # Add green corners
    for i in range(0, 3*rows, 3):
        for j in range(0, 3*cols, 3):
            has_non_zero = any(output_grid.get_cell(i+di, j+dj) not in [0, 1, 3] 
                               for di in range(3) for dj in range(3))
            if has_non_zero:
                for di, dj in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    if output_grid.get_cell(i+di, j+dj) == 0:
                        output_grid.set_cell(i+di, j+dj, 3)
    
    return output_grid
