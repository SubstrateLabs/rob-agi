from rob_agi.colored_grid import ColoredGrid

def solve_759f3fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the green cross and filling each quadrant
    with a fractal-like pattern of yellow and black squares.
    
    The pattern follows these rules:
    1. The green (3) cross is preserved from the input.
    2. Each quadrant is filled independently but symmetrically.
    3. Odd-numbered columns and rows from the cross are filled with yellow (4).
    4. Even-numbered columns and rows alternate between black (0) and yellow (4).
    5. The outer corners of each quadrant (furthest from the cross) are always yellow.
    
    This creates a nested square pattern in each quadrant that gets smaller
    towards the corners, while maintaining symmetry across the entire grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find the green cross
    cross_row = next(r for r in range(rows) if 3 in input_grid.values[r])
    cross_col = next(c for c in range(cols) if input_grid.values[cross_row][c] == 3)
    
    # Copy the green cross
    for r in range(rows):
        output_grid.values[r][cross_col] = 3
    for c in range(cols):
        output_grid.values[cross_row][c] = 3
    
    # Fill each quadrant
    quadrants = [
        (0, 0, cross_row, cross_col),
        (0, cross_col, cross_row, cols),
        (cross_row, 0, rows, cross_col),
        (cross_row, cross_col, rows, cols)
    ]
    
    for top, left, bottom, right in quadrants:
        for r in range(top, bottom):
            for c in range(left, right):
                if output_grid.values[r][c] == 3:
                    continue
                
                row_dist = min(abs(r - cross_row), abs(r - (bottom - 1)))
                col_dist = min(abs(c - cross_col), abs(c - (right - 1)))
                
                if row_dist % 2 == 1 or col_dist % 2 == 1:
                    output_grid.values[r][c] = 4
                elif row_dist == 0 or col_dist == 0:
                    output_grid.values[r][c] = 0
                else:
                    output_grid.values[r][c] = 4 if (row_dist + col_dist) % 2 == 0 else 0
    
    return output_grid
