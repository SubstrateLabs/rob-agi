from rob_agi.colored_grid import ColoredGrid

def solve_759f3fd3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the green cross and filling each quadrant
    with a staircase-like pattern of yellow squares.
    
    The pattern follows these rules:
    1. The green (3) cross is preserved from the input.
    2. Each quadrant is filled with a yellow (4) staircase pattern, starting from the corner adjacent to the cross.
    3. The edges of the grid are filled with a specific pattern of yellow (4) and black (0).
    4. The corners of the grid (except where the cross intersects) are always yellow (4).
    5. The staircase pattern in each quadrant follows a specific rule, creating nested rectangles.
    
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
    
    # Fill the edges
    for c in range(cols):
        if c != cross_col:
            output_grid.values[0][c] = 4  # Top edge
            output_grid.values[-1][c] = 4 if c % 2 == 0 else 0  # Bottom edge
    
    for r in range(rows):
        if r != cross_row:
            output_grid.values[r][0] = 4 if r % 2 == 0 else 0  # Left edge
            output_grid.values[r][-1] = 4 if r % 2 == 0 else 0  # Right edge
    
    # Fill each quadrant
    quadrants = [
        (0, 0, cross_row, cross_col),
        (0, cross_col + 1, cross_row, cols),
        (cross_row + 1, 0, rows, cross_col),
        (cross_row + 1, cross_col + 1, rows, cols)
    ]
    
    for top, left, bottom, right in quadrants:
        r, c = top, left
        step = 1
        while r < bottom and c < right:
            for i in range(step):
                if r + i < bottom and c < right:
                    output_grid.values[r + i][c] = 4
                if r < bottom and c + i < right:
                    output_grid.values[r][c + i] = 4
            r += 1
            c += 1
            step += 1
    
    return output_grid
