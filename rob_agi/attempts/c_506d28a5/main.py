from rob_agi.colored_grid import ColoredGrid

def solve_506d28a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by following these steps:
    1. Extracts the top 4 rows of the input grid.
    2. Creates a new 4x5 grid.
    3. For each column:
       - If there's any red cell in the column, all cells become green except:
         - If the top cell is black, it remains black.
         - If the bottom cell is black, it remains black.
       - If there's no red cell, the column remains unchanged.
    4. Returns the transformed grid as a new ColoredGrid object.
    """
    # Extract top 4 rows
    input_top = input_grid.values[:4]
    
    # Initialize output grid
    output = [[0 for _ in range(5)] for _ in range(4)]
    
    # Process each column
    for col in range(5):
        input_column = [input_top[row][col] for row in range(4)]
        
        # Check for red
        has_red = 2 in input_column
        
        if has_red:
            # Set column to green
            output_column = [3, 3, 3, 3]
            
            # Check first and last cells
            if input_column[0] == 0:
                output_column[0] = 0
            if input_column[3] == 0:
                output_column[3] = 0
        else:
            # Copy input column
            output_column = input_column
        
        # Add column to output
        for row in range(4):
            output[row][col] = output_column[row]
    
    return ColoredGrid(values=output)
