from rob_agi.colored_grid import ColoredGrid

def solve_cad67732(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size and extending the pattern.
    
    The function:
    1. Creates a new grid double the size of the input.
    2. Copies the input pattern to the top-left quadrant.
    3. Extends the pattern diagonally from bottom-right to top-left in the new space.
    4. Uses modular arithmetic to wrap around and continue the pattern seamlessly.
    
    This approach preserves and extends existing patterns, handling
    diagonal, checkerboard, and other complex arrangements.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(2*width)] for _ in range(2*height)])
    
    # Copy input grid to top-left quadrant
    for row in range(height):
        for col in range(width):
            new_grid.values[row][col] = input_grid.values[row][col]
    
    # Fill in the expanded areas of the new grid
    for row in range(2*height):
        for col in range(2*width):
            if row >= height or col >= width:
                src_row = (row - height) % height
                src_col = (col - width) % width
                new_grid.values[row][col] = new_grid.values[src_row][src_col]
    
    return new_grid
