from rob_agi.colored_grid import ColoredGrid

def solve_6f8cd79b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by adding a border of 8's (sky color) around the edge,
    while keeping the interior values unchanged.
    
    The border is always one cell thick, regardless of the input grid size.
    """
    height, width = input_grid.get_dimensions()
    
    # Create a new grid with the same dimensions
    output = input_grid.deep_copy()
    
    # Set the border cells to 8 (sky color)
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                output.set_cell(i, j, 8)
    
    return output
