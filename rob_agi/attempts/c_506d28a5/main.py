from rob_agi.colored_grid import ColoredGrid

def solve_506d28a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by following these steps:
    1. Extracts the top 4 rows of the input grid.
    2. Creates a new 4x5 grid.
    3. For each column:
       - If any cell in the column is non-black, the top and bottom cells become green.
       - For the middle two cells:
         - If there's any red cell in the column, both cells become green.
         - Otherwise, preserves the original colors from the input.
    4. Returns the transformed grid as a new ColoredGrid object.
    """
    def process_column(input_column):
        # For top and bottom rows
        top_bottom = 3 if any(cell != 0 for cell in input_column) else 0
        
        # For middle rows
        if any(cell == 2 for cell in input_column):
            middle = [3, 3]
        else:
            middle = [input_column[1], input_column[2]]
        
        return [top_bottom] + middle + [top_bottom]

    # Extract top 4 rows
    input_top = input_grid.values[:4]
    
    # Process each column
    output = []
    for col in range(5):
        input_column = [input_top[row][col] for row in range(4)]
        output_column = process_column(input_column)
        output.append(output_column)

    # Transpose the result to get rows instead of columns
    output = list(map(list, zip(*output)))
    
    return ColoredGrid(values=output)
