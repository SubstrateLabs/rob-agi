from rob_agi.colored_grid import ColoredGrid

def solve_e133d23d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 7x3 input grid into a 3x3 output grid based on color patterns.
    
    For each row in the input:
    - Divides the row into three sections: columns 0-1, 2-4, and 5-6
    - For each section, checks if it contains any of the colors: magenta (6), yellow (4), or sky blue (8)
    - If a section has any of these colors, sets the corresponding output cell to red (2)
    - Otherwise, sets the corresponding output cell to black (0)
    
    Returns the resulting 3x3 ColoredGrid.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(3)] for _ in range(3)])
    target_colors = {6, 4, 8}
    
    for row in range(3):
        input_row = input_grid.values[row]
        
        # Process section 1 (columns 0-1)
        output_grid.values[row][0] = 2 if any(color in target_colors for color in input_row[0:2]) else 0
        
        # Process section 2 (columns 2-4)
        output_grid.values[row][1] = 2 if any(color in target_colors for color in input_row[2:5]) else 0
        
        # Process section 3 (columns 5-6)
        output_grid.values[row][2] = 2 if any(color in target_colors for color in input_row[5:7]) else 0
    
    return output_grid
