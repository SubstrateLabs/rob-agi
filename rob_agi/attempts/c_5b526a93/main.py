from rob_agi.colored_grid import ColoredGrid

def solve_5b526a93(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying 3x3 blue square patterns and changing them to sky blue,
    except for patterns in the topmost and bottommost rows of patterns which remain unchanged.
    
    The function looks for 3x3 regions where:
    - The corners and center are blue (1)
    - The middle of each side is black (0)
    
    When such a pattern is found (except in the topmost and bottommost rows of patterns), it's changed to sky blue (8).
    Additionally, two more identical sky blue patterns are added in the same row at columns 6-8 and 12-14,
    unless there's an existing blue pattern in those positions.
    
    :param input_grid: The input ColoredGrid
    :return: The transformed ColoredGrid
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_blue_pattern(grid, row, col):
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

    patterns = []
    for row in range(rows - 2):
        for col in range(cols - 2):
            if is_blue_pattern(grid, row, col):
                patterns.append((row, col))

    if patterns:
        top_row = min(row for row, _ in patterns)
        bottom_row = max(row for row, _ in patterns)

        for row, col in patterns:
            if row != top_row and row != bottom_row:
                # Transform original pattern
                for i in range(3):
                    for j in range(3):
                        if (i + j) % 2 == 0:  # corners and center
                            grid.set_cell(row + i, col + j, 8)
                
                # Add two more patterns
                for new_col in [6, 12]:
                    if all(grid.get_cell(row + i, new_col + j) != 1 for i in range(3) for j in range(3)):
                        for i in range(3):
                            for j in range(3):
                                if (i + j) % 2 == 0:
                                    grid.set_cell(row + i, new_col + j, 8)
                                else:
                                    grid.set_cell(row + i, new_col + j, 0)

    return grid
