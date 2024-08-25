from rob_agi.colored_grid import ColoredGrid

def solve_8be77c9e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its height and creating a vertical mirror effect.
    
    The transformation follows these rules:
    1. The output grid is twice the height of the input grid.
    2. The top half of the output grid is identical to the input grid.
    3. The bottom half is a vertical mirror of the top half, with rows in reverse order.
    4. The very last row of the output grid is identical to the first row of the input grid.
    """
    # Get the dimensions of the input grid
    height, width = input_grid.get_dimensions()
    
    # Create a new grid with double the height
    output = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(2 * height)])
    
    # Copy the input grid to the top half of the output
    for i in range(height):
        for j in range(width):
            output.set_cell(i, j, input_grid.get_cell(i, j))
    
    # Add the mirrored bottom half
    for i in range(height - 1):  # Exclude the last row of the input grid
        for j in range(width):
            output.set_cell(2 * height - 2 - i, j, input_grid.get_cell(i + 1, j))
    
    # Set the last row to be identical to the first row of the input grid
    for j in range(width):
        output.set_cell(2 * height - 1, j, input_grid.get_cell(0, j))
    
    return output
