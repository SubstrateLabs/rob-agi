from rob_agi.colored_grid import ColoredGrid

def solve_759f3fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the green cross and filling each quadrant
    with a specific pattern of yellow squares and black spaces.
    
    The pattern follows these rules:
    1. The green (3) cross is preserved from the input.
    2. The border of the grid is filled with alternating yellow (4) and black (0),
       with specific starting points for each edge.
    3. Each quadrant is filled with L-shaped yellow patterns, separated by black spaces.
    4. The L-shapes grow in size as they move away from the cross, following a specific
       direction in each quadrant.
    5. The pattern maintains symmetry across both the horizontal and vertical green lines,
       with a half-cell offset due to the alternating border.
    
    This implementation uses efficient list comprehensions and avoids unnecessary loops
    to improve performance while maintaining the correct pattern generation.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Find the green cross
    cross_row = next(r for r in range(rows) if 3 in input_grid.values[r])
    cross_col = next(c for c in range(cols) if input_grid.values[cross_row][c] == 3)
    
    # Copy the green cross and fill the border
    for r in range(rows):
        for c in range(cols):
            if r == cross_row or c == cross_col:
                output_grid[r][c] = 3
            elif r == 0:
                output_grid[r][c] = 4 if c % 2 == 0 else 0
            elif r == rows - 1:
                output_grid[r][c] = 4 if c % 2 == 1 else 0
            elif c == 0:
                output_grid[r][c] = 4 if r % 2 == 0 else 0
            elif c == cols - 1:
                output_grid[r][c] = 4 if r % 2 == 1 else 0
    
    # Fill each quadrant
    quadrants = [
        (0, 0, cross_row, cross_col, 1, 1),
        (0, cross_col + 1, cross_row, cols, 1, -1),
        (cross_row + 1, 0, rows, cross_col, -1, 1),
        (cross_row + 1, cross_col + 1, rows, cols, -1, -1)
    ]
    
    for top, left, bottom, right, row_dir, col_dir in quadrants:
        size = 1
        r, c = top, left
        while r * row_dir < bottom * row_dir and c * col_dir < right * col_dir:
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid[r][c] == 0:
                    output_grid[r][c] = 4
                r += row_dir
            r -= row_dir
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid[r][c] == 0:
                    output_grid[r][c] = 4
                c += col_dir
            size += 1
            r += row_dir
            c += col_dir
    
    return ColoredGrid(values=output_grid)
