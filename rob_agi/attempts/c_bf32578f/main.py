from rob_agi.colored_grid import ColoredGrid

def solve_bf32578f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid pattern into a centered 4x4 shape.
    
    The function identifies the non-zero color in the input grid,
    creates a 4x4 shape (square or diamond) with that color,
    and places it in the center of a new grid of the same size
    as the input. The shape is a square for rectangular patterns
    and a diamond for non-rectangular patterns.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with the transformed pattern.
    """
    # Step 1: Analyze the input grid
    color = next(cell for row in input_grid.values for cell in row if cell != 0)
    is_rectangular = all(cell in (0, color) for row in input_grid.values for cell in row)
    
    # Step 2: Create the output shape
    shape = [[color] * 4 for _ in range(4)]
    if not is_rectangular:
        shape[0][0] = shape[0][3] = shape[3][0] = shape[3][3] = 0
    
    # Step 3 & 4: Prepare the output grid and place the shape
    output = input_grid.deep_copy()
    rows, cols = output.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            output.values[r][c] = 0
    
    top = (rows - 4) // 2
    left = (cols - 4) // 2
    
    for r in range(4):
        for c in range(4):
            if 0 <= top + r < rows and 0 <= left + c < cols:
                output.values[top + r][left + c] = shape[r][c]
    
    return output
