from rob_agi.colored_grid import ColoredGrid

def solve_a416b8f3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by duplicating each row horizontally.
    
    This function takes an input ColoredGrid and creates a new ColoredGrid
    where each row of the input is repeated twice horizontally. The resulting
    grid has the same number of rows as the input, but twice the number of columns.
    """
    # Extract the values from the input ColoredGrid
    input_values = input_grid.values
    
    # Apply the transformation: duplicate each row
    output_values = [row * 2 for row in input_values]
    
    # Create and return a new ColoredGrid with the transformed values
    return ColoredGrid(values=output_values)
