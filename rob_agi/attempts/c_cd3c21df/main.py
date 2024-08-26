from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique, largest subgrid pattern in the input grid.
    
    This function identifies the largest subgrid that appears only once in the input grid.
    If multiple such subgrids exist, it returns the one that appears first (top-left to bottom-right).
    
    The function scans the input grid from largest possible subgrid to smallest,
    checking if each subgrid is unique. It returns the first (top-left most) largest unique subgrid found.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique largest subgrid found, or None if no unique subgrid exists
    """
    rows, cols = input_grid.get_dimensions()
    
    def is_unique_subgrid(subgrid: ColoredGrid, orig_top: int, orig_left: int) -> bool:
        subgrid_height, subgrid_width = subgrid.get_dimensions()
        occurrence_count = 0
        
        for top in range(rows - subgrid_height + 1):
            for left in range(cols - subgrid_width + 1):
                if top == orig_top and left == orig_left:
                    occurrence_count += 1
                    continue
                
                if all(input_grid.values[top+i][left+j] == subgrid.values[i][j]
                       for i in range(subgrid_height)
                       for j in range(subgrid_width)):
                    occurrence_count += 1
                    
                if occurrence_count > 1:
                    return False
        
        return occurrence_count == 1

    for height in range(rows, 0, -1):
        for width in range(cols, 0, -1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_unique_subgrid(subgrid, top, left):
                        return subgrid
    
    return None
