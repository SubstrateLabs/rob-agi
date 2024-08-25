from rob_agi.colored_grid import ColoredGrid

def solve_ca8de6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 grid into a 3x3 grid by extracting specific elements.
    
    This function takes the corners, the first non-zero elements from each edge (excluding corners),
    and the center element from the input 5x5 grid and arranges them into a new 3x3 grid.
    
    The resulting 3x3 grid is structured as follows:
    - Top row: [top-left corner, first non-zero element from top row, top-right corner]
    - Middle row: [first non-zero element from left column, center, first non-zero element from right column]
    - Bottom row: [bottom-left corner, first non-zero element from bottom row, bottom-right corner]
    """
    # Extract corners
    top_left = input_grid.values[0][0]
    top_right = input_grid.values[0][4]
    bottom_left = input_grid.values[4][0]
    bottom_right = input_grid.values[4][4]
    
    # Find first non-zero element in top row (excluding corners)
    top_middle = next(val for val in input_grid.values[0][1:4] if val != 0)
    
    # Find first non-zero element in bottom row (excluding corners)
    bottom_middle = next(val for val in input_grid.values[4][1:4] if val != 0)
    
    # Find first non-zero element in leftmost column (excluding corners)
    left_middle = next(input_grid.values[i][0] for i in range(1, 4) if input_grid.values[i][0] != 0)
    
    # Find first non-zero element in rightmost column (excluding corners)
    right_middle = next(input_grid.values[i][4] for i in range(1, 4) if input_grid.values[i][4] != 0)
    
    # Extract center
    center = input_grid.values[2][2]
    
    # Create new 3x3 grid
    new_values = [
        [top_left, top_middle, top_right],
        [left_middle, center, right_middle],
        [bottom_left, bottom_middle, bottom_right]
    ]
    
    return ColoredGrid(values=new_values)
