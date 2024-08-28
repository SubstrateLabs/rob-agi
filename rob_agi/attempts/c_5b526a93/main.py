from rob_agi.colored_grid import ColoredGrid

def solve_5b526a93(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying 3x3 blue square patterns and changing them to sky blue,
    except for the leftmost occurrence in each row which remains unchanged.
    
    The function looks for 3x3 regions where:
    - The corners and center are blue (1)
    - The middle of each side is black (0)
    
    When such a pattern is found (except for the leftmost in each row), it's changed to sky blue (8).
    Additionally, new sky blue patterns are added in the same row maintaining the spacing between patterns,
    if there's available space. The bottom row of patterns is always preserved as blue.
    
    :param input_grid: The input ColoredGrid
    :return: The transformed ColoredGrid
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_blue_pattern(row: int, col: int) -> bool:
        pattern = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        return all(grid.get_cell(row + i, col + j) == pattern[i][j] 
                   for i in range(3) for j in range(3))

    def transform_to_sky_blue(row: int, col: int) -> None:
        for i in [0, 2]:
            for j in [0, 2]:
                grid.set_cell(row + i, col + j, 8)
        grid.set_cell(row + 1, col + 1, 8)

    def add_sky_blue_pattern(row: int, col: int) -> None:
        transform_to_sky_blue(row, col)

    for row in range(rows - 2):
        pattern_positions = []
        for col in range(cols - 2):
            if is_blue_pattern(row, col):
                pattern_positions.append(col)

        if pattern_positions:
            # Skip the first (leftmost) pattern and preserve the bottom row
            if row < rows - 3:  # Not the bottom row of patterns
                for col in pattern_positions[1:]:
                    transform_to_sky_blue(row, col)

                # Add new patterns
                last_col = pattern_positions[-1]
                pattern_width = 3
                space_between = pattern_positions[1] - pattern_positions[0] if len(pattern_positions) > 1 else pattern_width + 1

                next_col = last_col + space_between
                while next_col + pattern_width <= cols:
                    if all(grid.get_cell(row + i, next_col + j) == 0 for i in range(3) for j in range(3)):
                        add_sky_blue_pattern(row, next_col)
                    next_col += space_between

    return grid
