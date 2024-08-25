from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_9a4bb226(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Finds a 3x3 subgrid within the input grid that contains exactly three colors,
    where one color appears 5 times and the other two colors appear 2 times each.
    
    The function checks four specific 3x3 regions in the input grid:
    top-left, top-right, bottom-left, and bottom-right.
    It returns the first valid subgrid found as a new ColoredGrid.
    
    If no valid subgrid is found, returns None.
    """
    def is_valid_subgrid(subgrid: List[List[int]]) -> bool:
        color_counts = {}
        for row in subgrid:
            for color in row:
                if color != 0:
                    color_counts[color] = color_counts.get(color, 0) + 1
        return len(color_counts) == 3 and sorted(color_counts.values()) == [2, 2, 5]

    regions = [
        [input_grid.values[i][j:j+3] for i in range(2, 5)] for j in [2, 8]
    ] + [
        [input_grid.values[i][j:j+3] for i in range(7, 10)] for j in [2, 8]
    ]

    for region in regions:
        if is_valid_subgrid(region):
            return ColoredGrid(values=region)

    return None
