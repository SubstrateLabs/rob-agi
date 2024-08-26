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
    
    This creates a complex pattern that respects the green cross while filling the rest
    of the grid with a consistent yellow and black design.
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
    
    # Fill the border
    for c in range(cols):
        output_grid.values[0][c] = 4 if c % 2 == 0 else 0  # Top edge
        output_grid.values[-1][c] = 4 if c % 2 == 1 else 0  # Bottom edge
    for r in range(1, rows - 1):
        output_grid.values[r][0] = 4 if r % 2 == 0 else 0  # Left edge
        output_grid.values[r][-1] = 4 if r % 2 == 1 else 0  # Right edge
    
    # Fill each quadrant
    quadrants = [
        (0, 0, cross_row, cross_col, 1, 1),
        (0, cross_col + 1, cross_row, cols, 1, -1),
        (cross_row + 1, 0, rows, cross_col, -1, 1),
        (cross_row + 1, cross_col + 1, rows, cols, -1, -1)
    ]
    
    for top, left, bottom, right, row_dir, col_dir in quadrants:
        r, c = top, left
        size = 1
        while r != bottom and c != right:
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] != 3:
                    output_grid.values[r][c] = 4
                r += row_dir
            r -= row_dir
            for i in range(size):
                if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] != 3:
                    output_grid.values[r][c] = 4
                c += col_dir
            c -= col_dir
            size += 1
            r += row_dir
            c += col_dir
    
    return output_grid
