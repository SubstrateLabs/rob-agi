from rob_agi.colored_grid import ColoredGrid

def solve_759f3fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the green cross and filling each quadrant
    with a staircase-like pattern of yellow squares.
    
    The pattern follows these rules:
    1. The green (3) cross is preserved from the input.
    2. Each quadrant is filled with a yellow (4) staircase pattern, starting from the corner adjacent to the cross.
    3. The edges of the grid are filled with alternating yellow (4) and black (0), with specific rules for each edge.
    4. The corners of the grid (except where the cross intersects) are always yellow (4).
    
    This creates a complex pattern that respects the green cross while filling the rest of the grid
    with a consistent yellow and black design.
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
        (0, cross_col + 1, cross_row, cols),
        (cross_row + 1, 0, rows, cross_col),
        (cross_row + 1, cross_col + 1, rows, cols)
    ]
    
    for top, left, bottom, right in quadrants:
        step_width, step_length = 2, 1
        r, c = top, left
        while r < bottom and c < right:
            for i in range(step_length):
                if r + i < bottom and c < right:
                    output_grid.values[r + i][c] = 4
            r += step_length
            c += 1
            step_length += 1
            if step_length > step_width:
                step_width += 2
                step_length = 1
    
    # Fill the edges
    for c in range(cols):
        if c != cross_col:
            output_grid.values[0][c] = 4  # Top edge
            output_grid.values[-1][c] = 4 if c % 2 == 0 else 0  # Bottom edge
    
    for r in range(1, rows - 1):
        if r != cross_row:
            output_grid.values[r][0] = 4 if r % 2 == 0 else 0  # Left edge
            output_grid.values[r][-1] = 4 if r % 2 == 0 else 0  # Right edge
    
    # Ensure corners are yellow (except where cross intersects)
    if cross_row != 0 and cross_col != 0:
        output_grid.values[0][0] = 4
    if cross_row != 0 and cross_col != cols - 1:
        output_grid.values[0][-1] = 4
    if cross_row != rows - 1 and cross_col != 0:
        output_grid.values[-1][0] = 4
    if cross_row != rows - 1 and cross_col != cols - 1:
        output_grid.values[-1][-1] = 4
    
    return output_grid
