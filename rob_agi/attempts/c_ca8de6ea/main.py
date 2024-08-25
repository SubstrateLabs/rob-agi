from rob_agi.colored_grid import ColoredGrid

def solve_ca8de6ea(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 5x5 grid into a 3x3 grid by extracting specific elements.
    
    This function takes the corners, the elements at specific positions,
    and the center element from the input 5x5 grid and arranges them into a new 3x3 grid.
    
    The resulting 3x3 grid is structured as follows:
    - Top row: [top-left corner, element at (1,1), top-right corner]
    - Middle row: [element at (2,1), center, element at (2,3)]
    - Bottom row: [bottom-left corner, element at (3,1), bottom-right corner]
    """
    # Extract corners
    top_left = input_grid.values[0][0]
    top_right = input_grid.values[0][4]
    bottom_left = input_grid.values[4][0]
    bottom_right = input_grid.values[4][4]
    
    # Extract middle elements
    top_middle = input_grid.values[1][1]
    right_middle = input_grid.values[2][3]
    bottom_middle = input_grid.values[3][1]
    left_middle = input_grid.values[2][1]
    
    # Extract center
    center = input_grid.values[2][2]
    
    # Create new 3x3 grid
    new_values = [
        [top_left, top_middle, top_right],
        [left_middle, center, right_middle],
        [bottom_left, bottom_middle, bottom_right]
    ]
    
    return ColoredGrid(values=new_values)
