from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique, solid color block with the highest structural significance in the input grid.
    
    This function scans the input grid for all possible subgrids, identifying unique solid color blocks.
    It prioritizes larger blocks and those that are not black (empty space). The function returns the
    solid color block that is both unique within the grid and has the highest structural significance.
    
    The structural significance is determined by the size of the block and its color, with a preference
    for non-black colors and larger sizes.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique solid color block with the highest structural significance
    """
    rows, cols = input_grid.get_dimensions()
    
    def is_solid_color_block(subgrid: ColoredGrid) -> bool:
        color = next((c for row in subgrid.values for c in row if c != 0), None)
        return color is not None and all(all(c in (0, color) for c in row) for row in subgrid.values)
    
    def is_unique_subgrid(subgrid: ColoredGrid, orig_top: int, orig_left: int) -> bool:
        subgrid_height, subgrid_width = subgrid.get_dimensions()
        for top in range(rows - subgrid_height + 1):
            for left in range(cols - subgrid_width + 1):
                if top == orig_top and left == orig_left:
                    continue
                if all(input_grid.values[top+i][left+j] == subgrid.values[i][j]
                       for i in range(subgrid_height)
                       for j in range(subgrid_width)):
                    return False
        return True

    def calculate_significance(subgrid: ColoredGrid) -> float:
        height, width = subgrid.get_dimensions()
        color = next(c for row in subgrid.values for c in row if c != 0)
        return height * width * (color + 1)  # Prioritize non-black colors and larger sizes

    best_subgrid = None
    best_score = float('-inf')

    for height in range(1, rows + 1):
        for width in range(1, cols + 1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_solid_color_block(subgrid) and is_unique_subgrid(subgrid, top, left):
                        score = calculate_significance(subgrid)
                        if score > best_score:
                            best_score = score
                            best_subgrid = subgrid

    return best_subgrid
