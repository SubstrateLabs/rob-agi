from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell.
    
    For each non-zero cell in the input:
    - Place the value in the center of the corresponding 3x3 subgrid in the output
    - Place the value in the appropriate corners of the 9x9 output grid
    
    This creates a pattern where non-zero values from the input appear
    in the center of subgrids and form a frame around the entire output grid,
    while preserving the overall structure of the input grid.
    """
    # Create a new 9x9 grid filled with zeros
    output_values = [[0 for _ in range(9)] for _ in range(9)]
    
    # Iterate through each cell of the input grid
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            
            if v != 0:
                # Set the center of the corresponding 3x3 subgrid
                center_row = 3 * i + 1
                center_col = 3 * j + 1
                output_values[center_row][center_col] = v
                
                # Set the appropriate corners of the 9x9 output grid
                if i == 0:
                    output_values[0][3*j] = v
                    output_values[0][3*j+2] = v
                if i == 2:
                    output_values[8][3*j] = v
                    output_values[8][3*j+2] = v
                if j == 0:
                    output_values[3*i][0] = v
                    output_values[3*i+2][0] = v
                if j == 2:
                    output_values[3*i][8] = v
                    output_values[3*i+2][8] = v
    
    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output_values)
