from rob_agi.colored_grid import ColoredGrid

def solve_5b526a93(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying 3x3 blue square patterns and changing them to sky blue,
    except for patterns in the bottom three rows which remain unchanged.
    
    The function looks for 3x3 regions where:
    - The corners and center are blue (1)
    - The middle of each side is black (0)
    
    When such a pattern is found (except in the bottom three rows), it's changed to sky blue (8).
    All transformations are applied simultaneously after identifying all patterns.
    
    :param input_grid: The input ColoredGrid
    :return: The transformed ColoredGrid
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_pattern_match(grid, row, col):
        pattern = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ]
        for i in range(3):
            for j in range(3):
                if grid.get_cell(row + i, col + j) != pattern[i][j]:
                    return False
        return True

    transform_positions = []

    for row in range(rows - 2):
        for col in range(cols - 2):
            if is_pattern_match(grid, row, col) and row < rows - 3:
                transform_positions.append((row, col))

    for row, col in transform_positions:
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if (i, j) != (1, 1) and (i + j) % 2 == 0:
                    grid.set_cell(row + i, col + j, 8)

    return grid
