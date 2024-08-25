from rob_agi.colored_grid import ColoredGrid

def solve_332efdb3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a pattern where:
    1. All odd-indexed rows (1-indexed) are filled with blue (1).
    2. Even-indexed rows alternate between blue (1) and black (0), starting and ending with blue.
    3. The outer border is always blue (1).
    """
    size = len(input_grid.values)
    new_values = []
    
    for i in range(size):
        if i % 2 == 0:  # Odd-indexed row (0-indexed in code, but 1-indexed in problem description)
            new_values.append([1] * size)
        else:  # Even-indexed row
            new_values.append([1 if j % 2 == 0 else 0 for j in range(size)])
    
    return ColoredGrid(values=new_values)
