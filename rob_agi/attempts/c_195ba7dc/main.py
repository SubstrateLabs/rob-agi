from rob_agi.colored_grid import ColoredGrid

def solve_195ba7dc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 5x13 input grid into a 5x6 output grid based on the presence of orange (7) cells.
    
    The function divides the input grid into left and right sections, ignoring the middle column.
    For each pair of columns in these sections, if an orange cell is present, the corresponding
    output cell is set to blue (1). Otherwise, it remains black (0).
    
    This results in a compressed representation of the input, where orange areas become blue,
    and all-black areas remain black.
    """
    # Initialize output grid (5x6) filled with zeros
    output_values = [[0 for _ in range(6)] for _ in range(5)]
    
    # Process each row
    for row in range(5):
        # Process left section
        if 7 in input_grid.values[row][0:2]:
            output_values[row][0] = 1
        if 7 in input_grid.values[row][2:4]:
            output_values[row][1] = 1
        if 7 in input_grid.values[row][4:6]:
            output_values[row][2] = 1
        
        # Process right section
        if 7 in input_grid.values[row][7:9]:
            output_values[row][3] = 1
        if 7 in input_grid.values[row][9:11]:
            output_values[row][4] = 1
        if 7 in input_grid.values[row][11:13]:
            output_values[row][5] = 1
    
    # Create and return the output ColoredGrid
    return ColoredGrid(values=output_values)
