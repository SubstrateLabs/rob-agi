from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5b526a93(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying 3x3 blue square patterns and changing them to sky blue,
    except for the first occurrence of the pattern which remains unchanged.
    
    The function looks for 3x3 regions where:
    - The corners and center are blue (1)
    - The middle of each side is black (0)
    
    When such a pattern is found (except for the first occurrence), it's changed to sky blue (8).
    Additionally, two more identical sky blue patterns are added in the same row at columns 6-8 and 12-14,
    if there's available space.
    
    :param input_grid: The input ColoredGrid
    :return: The transformed ColoredGrid
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_blue_pattern(grid: ColoredGrid, row: int, col: int) -> bool:
        pattern = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        return all(grid.get_cell(row + i, col + j) == pattern[i][j] 
                   for i in range(3) for j in range(3))

    def find_blue_patterns(grid: ColoredGrid) -> List[Tuple[int, int]]:
        return [(row, col) for row in range(rows - 2) 
                for col in range(cols - 2) if is_blue_pattern(grid, row, col)]

    def transform_to_sky_blue(grid: ColoredGrid, row: int, col: int) -> None:
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:  # corners and center
                    grid.set_cell(row + i, col + j, 8)

    patterns = find_blue_patterns(grid)
    patterns.sort(key=lambda x: (x[0], x[1]))  # Sort by row, then column

    first_pattern_processed = False
    for row, col in patterns:
        if not first_pattern_processed:
            first_pattern_processed = True
            continue
        
        transform_to_sky_blue(grid, row, col)
        
        for new_col in [6, 12]:
            if new_col + 2 < cols and all(grid.get_cell(row + i, new_col + j) == 0 for i in range(3) for j in range(3)):
                transform_to_sky_blue(grid, row, new_col)

    return grid
