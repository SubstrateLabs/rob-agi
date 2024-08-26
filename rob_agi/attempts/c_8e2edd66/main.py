from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell.
    
    For each non-zero cell in the input:
    - Create a 3x3 subgrid in the output
    - Place the input cell's value in the four corners of this subgrid
    - Leave the center and edges of the subgrid as zeros
    
    Zero cells in the input result in 3x3 subgrids of all zeros in the output.
    """
    # Create a new 9x9 grid filled with zeros
    output_values = [[0 for _ in range(9)] for _ in range(9)]
    
    # Iterate through each cell of the input grid
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            
            if v != 0:
                # Calculate the top-left corner of the corresponding 3x3 subgrid
                row = 3 * i
                col = 3 * j
                
                # Set the four corners of the 3x3 subgrid
                output_values[row][col] = v
                output_values[row][col+2] = v
                output_values[row+2][col] = v
                output_values[row+2][col+2] = v
    
    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output_values)
