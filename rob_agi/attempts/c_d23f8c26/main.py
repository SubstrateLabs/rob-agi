from rob_agi.colored_grid import ColoredGrid

def solve_d23f8c26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d23f8c26 challenge by keeping only the middle column of the input grid
    and the values that appear more than once in the entire grid or are 0 (black).
    
    1. Determine the dimensions of the input grid.
    2. Calculate the middle column index.
    3. Create a new ColoredGrid with the same dimensions as the input, filled with zeros.
    4. Count the occurrences of each value in the entire grid.
    5. For each row, copy the value from the middle column to the result grid
       only if it appears more than once in the entire grid or if it's 0 (black).
    6. Set all other columns to 0 (black).
    """
    # Get the dimensions of the input grid
    height, width = input_grid.get_dimensions()
    
    # Calculate the middle column index
    middle_col = width // 2
    
    # Create a new ColoredGrid with the same dimensions as the input, filled with zeros
    result = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    # Count occurrences of each value in the entire grid
    value_counts = {}
    for row in range(height):
        for col in range(width):
            value = input_grid.get_cell(row, col)
            value_counts[value] = value_counts.get(value, 0) + 1
    
    # Process each row
    for row in range(height):
        # Get the value from the middle column
        middle_value = input_grid.get_cell(row, middle_col)
        
        # If the middle value is 0 or appears more than once in the entire grid, keep it in the result
        if middle_value == 0 or value_counts[middle_value] > 1:
            result.set_cell(row, middle_col, middle_value)
    
    return result
