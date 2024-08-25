from rob_agi.colored_grid import ColoredGrid

def solve_a9f96cdd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing the '2' with '0' and placing four new numbers (3, 6, 7, 8) around it.
    The new numbers are placed in the following positions relative to the original '2':
    - '3': One row up, one column left
    - '6': One row up, one column right
    - '7': One row down, one column right
    - '8': One row down, one column left
    If a position is outside the grid boundaries, that number is not placed.
    """
    def find_2(grid):
        for i, row in enumerate(grid.values):
            for j, val in enumerate(row):
                if val == 2:
                    return i, j
        return None

    def is_valid_position(grid, row, col):
        return 0 <= row < len(grid.values) and 0 <= col < len(grid.values[0])

    output = input_grid.deep_copy()
    pos = find_2(input_grid)
    
    if pos:
        row, col = pos
        output.set_cell(row, col, 0)
        
        transformations = [
            (-1, -1, 3), (-1, 1, 6),
            (1, 1, 7), (1, -1, 8)
        ]
        
        for dr, dc, value in transformations:
            new_row, new_col = row + dr, col + dc
            if is_valid_position(output, new_row, new_col):
                output.set_cell(new_row, new_col, value)
    
    return output
