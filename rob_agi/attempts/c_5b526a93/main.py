from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5b526a93(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying 3x3 blue square patterns and changing them to sky blue,
    except for patterns in the first row of patterns which remain unchanged.
    
    The function looks for 3x3 regions where:
    - The corners and center are blue (1)
    - The middle of each side is black (0)
    
    When such a pattern is found (except in the first row of patterns), it's changed to sky blue (8).
    Additionally, two more identical sky blue patterns are added in the same row at columns 6-8 and 12-14,
    if there's available space.
    
    If there are gaps between rows with patterns, new rows with three sky blue patterns are added,
    aligned with the topmost group.
    
    :param input_grid: The input ColoredGrid
    :return: The transformed ColoredGrid
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_blue_pattern(grid: ColoredGrid, row: int, col: int) -> bool:
        pattern = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        return all(grid.get_cell(row + i, col + j) == pattern[i][j] 
                   for i in range(3) for j in range(3))

    def find_patterns(grid: ColoredGrid) -> List[Tuple[int, int]]:
        return [(row, col) for row in range(rows - 2) 
                for col in range(cols - 2) if is_blue_pattern(grid, row, col)]

    def group_patterns(patterns: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        patterns.sort(key=lambda x: x[0])
        groups = []
        for row, group in itertools.groupby(patterns, key=lambda x: x[0]):
            groups.append(list(group))
        return groups

    def transform_to_sky_blue(grid: ColoredGrid, row: int, col: int) -> None:
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:  # corners and center
                    grid.set_cell(row + i, col + j, 8)

    def add_sky_blue_pattern(grid: ColoredGrid, row: int, col: int) -> None:
        for i in range(3):
            for j in range(3):
                grid.set_cell(row + i, col + j, 8 if (i + j) % 2 == 0 else 0)

    patterns = find_patterns(grid)
    if not patterns:
        return grid

    pattern_groups = group_patterns(patterns)
    top_group = pattern_groups[0]
    top_row = top_group[0][0]

    for group in pattern_groups[1:]:
        row = group[0][0]
        for _, col in group:
            transform_to_sky_blue(grid, row, col)
        
        # Add additional patterns if needed
        existing_cols = [col for _, col in group]
        for new_col in [6, 12]:
            if new_col not in existing_cols and new_col + 2 < cols:
                add_sky_blue_pattern(grid, row, new_col)

    # Fill gaps between groups
    for i in range(1, len(pattern_groups)):
        prev_row = pattern_groups[i-1][0][0]
        curr_row = pattern_groups[i][0][0]
        if curr_row - prev_row > 3:
            new_row = prev_row + 3
            for col in [2, 6, 12]:
                if col + 2 < cols:
                    add_sky_blue_pattern(grid, new_row, col)

    return grid
