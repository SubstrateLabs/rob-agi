from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors upwards within each column.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. For each column, scan from bottom to top.
    3. Use a stack to keep track of non-black colors encountered.
    4. Fill all cells above with the top color on the stack.
    5. Add new non-black colors to the stack when encountered.
    
    This function preserves original color positions, extends colors upwards,
    maintains the order of colors when stacked, and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    for col in range(cols):
        color_stack = []
        for row in range(rows - 1, -1, -1):
            if result.values[row][col] != 0:
                color_stack.append(result.values[row][col])
            if color_stack:
                result.values[row][col] = color_stack[-1]
    
    return result
