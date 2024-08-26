from rob_agi.colored_grid import ColoredGrid

def solve_506d28a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by following these steps:
    1. Extracts the top 4 rows of the input grid.
    2. Creates a new 4x5 grid.
    3. For each column:
       - If there's any red cell in the column:
         - Fill the column with green.
         - Preserve contiguous black regions from the top and bottom edges.
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
            # Start with a green column
            output_column = [3, 3, 3, 3]
            
            # Preserve black from the top
            for row in range(4):
                if input_column[row] == 0:
                    output_column[row] = 0
                else:
                    break
            
            # Preserve black from the bottom
            for row in range(3, -1, -1):
                if input_column[row] == 0:
                    output_column[row] = 0
                else:
                    break
        else:
            # Copy input column if no red
            output_column = input_column
        
        # Add column to output
        for row in range(4):
            output[row][col] = output_column[row]
    
    return ColoredGrid(values=output)
