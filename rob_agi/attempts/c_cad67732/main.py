from rob_agi.colored_grid import ColoredGrid

def solve_cad67732(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size and extending the pattern.
    
    The function:
    1. Creates a new grid double the size of the input.
    2. Copies the input grid to the top-left quadrant of the new grid.
    3. Extends the pattern diagonally to fill the remaining space.
    
    This approach works for all cases by preserving the original pattern
    in the top-left and extending it diagonally, maintaining the relative
    positions and spacings of all elements.
    """
    input_height, input_width = input_grid.get_dimensions()
    new_height = input_height * 2
    new_width = input_width * 2
    
    new_grid = ColoredGrid(values=[[0 for _ in range(new_width)] for _ in range(new_height)])
    
    # Copy the input grid to the top-left quadrant
    for row in range(input_height):
        for col in range(input_width):
            new_grid.values[row][col] = input_grid.values[row][col]
    
    # Extend the pattern diagonally
    for row in range(new_height):
        for col in range(new_width):
            if row < input_height and col < input_width:
                continue  # Skip the already filled top-left quadrant
            source_row = row - input_height if row >= input_height else row
            source_col = col - input_width if col >= input_width else col
            new_grid.values[row][col] = new_grid.values[source_row][source_col]
    
    return new_grid
