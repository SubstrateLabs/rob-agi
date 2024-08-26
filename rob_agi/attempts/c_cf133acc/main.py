from rob_agi.colored_grid import ColoredGrid

def solve_cf133acc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending non-black colors upwards within each column.
    
    The function performs the following steps:
    1. Create a deep copy of the input grid.
    2. For each column, scan from bottom to top.
    3. Use a stack to keep track of non-black colors encountered.
    4. Fill black cells above with the top color on the stack.
    5. Add new non-black colors to the stack when encountered.
    6. Stop filling when a non-black cell is encountered.
    
    This function preserves original color positions, extends colors upwards until a non-black cell,
    maintains the order of colors when stacked, and works for grids of size 1x1 to 30x30.
    """
    result = input_grid.deep_copy()
    rows, cols = result.get_dimensions()
    
    for col in range(cols):
        color_stack = []
        for row in range(rows - 1, -1, -1):
            current_color = result.values[row][col]
            if current_color != 0:
                color_stack.append(current_color)
            elif color_stack:
                result.values[row][col] = color_stack[-1]
            else:
                break  # Stop filling when encountering a non-black cell with an empty stack
    
    return result
