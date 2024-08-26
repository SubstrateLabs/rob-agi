from rob_agi.colored_grid import ColoredGrid

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it horizontally while preserving color groups.
    
    The function identifies non-black color groups, moves them to the left side of the grid,
    and removes unnecessary black columns. This maintains the vertical structure and alignment
    of color groups while producing the narrowest possible output grid.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_col = 0

    def process_color_group(start, end):
        nonlocal current_col
        for col in range(start, end):
            for row in range(rows):
                new_grid[row][current_col] = input_grid.values[row][col]
            current_col += 1

    col = 0
    while col < cols:
        if any(input_grid.values[row][col] != 0 for row in range(rows)):
            start = col
            while col < cols and any(input_grid.values[row][col] != 0 for row in range(rows)):
                col += 1
            process_color_group(start, col)
        else:
            col += 1

    # Trim trailing black columns
    while all(new_grid[row][-1] == 0 for row in range(rows)):
        for row in range(rows):
            new_grid[row].pop()

    return ColoredGrid(values=new_grid)
