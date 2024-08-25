from rob_agi.colored_grid import ColoredGrid

def solve_fc754716(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by creating a border of the non-zero color around the entire grid.
    The border is one cell thick on all sides, and the interior is filled with zeros.
    """
    # Get dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Find the non-zero color in the input grid
    border_color = next((color for row in input_grid.values for color in row if color != 0), None)
    
    if border_color is None:
        raise ValueError("No non-zero color found in the input grid")
    
    # Create a new grid with the same dimensions
    output_values = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill the perimeter with the border color
    for i in range(rows):
        output_values[i][0] = border_color
        output_values[i][cols-1] = border_color
    for j in range(cols):
        output_values[0][j] = border_color
        output_values[rows-1][j] = border_color
    
    return ColoredGrid(values=output_values)
