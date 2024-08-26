from rob_agi.colored_grid import ColoredGrid

def solve_4852f2fa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the number of yellow squares.
    
    1. Count yellow (4) squares in the input grid.
    2. Create a 3xN output grid where N = (yellow_count + 1) * 3 - 1.
    3. Fill the top row with alternating [0, 8, 8] pattern.
    4. Fill the middle row entirely with 8's.
    5. Fill the bottom row with alternating [0, 8, 0] pattern.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Count yellow squares
    yellow_count = sum(row.count(4) for row in input_grid.values)
    
    # Calculate output width
    output_width = (yellow_count + 1) * 3 - 1
    
    # Create top row
    top_row = [0, 8, 8] * (yellow_count + 1)
    top_row.pop()  # Remove the last 0
    
    # Create middle row
    middle_row = [8] * output_width
    
    # Create bottom row
    bottom_row = [0, 8, 0] * (yellow_count + 1)
    bottom_row.pop()  # Remove the last 0
    
    # Combine rows
    output = [top_row, middle_row, bottom_row]
    
    return ColoredGrid(values=output)
