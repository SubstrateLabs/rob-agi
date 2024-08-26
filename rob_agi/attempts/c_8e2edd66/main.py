from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell.
    
    The transformation follows these rules:
    1. Place each non-zero value from the input grid in the center of the corresponding 3x3 subgrid in the output.
    2. For corner values in the input, place them in the corresponding corners of the output grid.
    3. For edge values in the input (not corners), place them in the middle of the corresponding edge in the output grid.
    4. All other positions in the output grid remain zero (black).

    This creates a pattern that preserves the structure of the input while expanding it into a larger grid.
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
                
                # Handle corners
                if (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                    output_row = 0 if i == 0 else 8
                    output_col = 0 if j == 0 else 8
                    output_values[output_row][output_col] = v
                
                # Handle edges
                elif i == 1 and j in [0, 2]:
                    output_col = 0 if j == 0 else 8
                    output_values[4][output_col] = v
                elif j == 1 and i in [0, 2]:
                    output_row = 0 if i == 0 else 8
                    output_values[output_row][4] = v
    
    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output_values)
