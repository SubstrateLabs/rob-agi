from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_9a4bb226(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Finds a 3x3 subgrid within the input grid that contains exactly three colors,
    where one color appears 5 times and the other two colors appear 2 times each.
    
    The function scans the input grid for all possible 3x3 subgrids, checks if they meet
    the criteria, and returns the first valid subgrid found as a new ColoredGrid.
    
    If no valid subgrid is found or if the input grid is too small, returns None.
    """
    def is_valid_subgrid(subgrid: List[List[int]]) -> bool:
        color_counts = {}
        for row in subgrid:
            for color in row:
                if color != 0:
                    color_counts[color] = color_counts.get(color, 0) + 1
        return len(color_counts) == 3 and sorted(color_counts.values()) == [2, 2, 5]

    rows, cols = input_grid.get_dimensions()
    
    if rows < 3 or cols < 3:
        return None

    for i in range(rows - 2):
        for j in range(cols - 2):
            subgrid = [row[j:j+3] for row in input_grid.values[i:i+3]]
            if is_valid_subgrid(subgrid):
                return ColoredGrid(values=subgrid)

    return None
