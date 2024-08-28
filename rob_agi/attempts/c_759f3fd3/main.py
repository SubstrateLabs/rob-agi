from rob_agi.colored_grid import ColoredGrid

def solve_759f3fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the green cross and filling each quadrant
    with a specific pattern of yellow squares and black spaces.
    
    The pattern follows these rules:
    1. The green (3) cross is preserved from the input.
    2. The border of the grid is filled with alternating yellow (4) and black (0),
       starting with yellow at the top-left and bottom-right corners.
    3. Each quadrant is filled with L-shaped yellow patterns, growing outward from the cross.
    4. The L-shapes alternate between yellow and black, increasing in size as they move away from the cross.
    5. The pattern maintains symmetry across both the horizontal and vertical green lines,
       with a half-cell offset due to the alternating border.
    6. The pattern adapts to different grid sizes and cross positions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Find the green cross
    cross_row = next(r for r in range(rows) if 3 in input_grid.values[r])
    cross_col = next(c for c in range(cols) if input_grid.values[cross_row][c] == 3)
    
    # Copy the green cross
    for r in range(rows):
        for c in range(cols):
            if r == cross_row or c == cross_col:
                output_grid[r][c] = 3
    
    # Fill the border
    for r in range(rows):
        output_grid[r][0] = 4 if r % 2 == 0 else 0
        output_grid[r][-1] = 4 if r % 2 == 1 else 0
    for c in range(cols):
        output_grid[0][c] = 4 if c % 2 == 0 else 0
        output_grid[-1][c] = 4 if c % 2 == 1 else 0
    
    def fill_quadrant(top, left, bottom, right, row_dir, col_dir):
        r, c = top, left
        size = 1
        while r * row_dir < bottom * row_dir and c * col_dir < right * col_dir:
            color = 4 if size % 2 == 1 else 0
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid[r][c] == 0:
                    output_grid[r][c] = color
                r += row_dir
            r -= row_dir
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid[r][c] == 0:
                    output_grid[r][c] = color
                c += col_dir
            size += 1
            r += row_dir
            c += col_dir
    
    # Fill each quadrant
    fill_quadrant(0, 0, cross_row, cross_col, 1, 1)
    fill_quadrant(0, cross_col + 1, cross_row, cols, 1, -1)
    fill_quadrant(cross_row + 1, 0, rows, cross_col, -1, 1)
    fill_quadrant(cross_row + 1, cross_col + 1, rows, cols, -1, -1)
    
    return ColoredGrid(values=output_grid)
