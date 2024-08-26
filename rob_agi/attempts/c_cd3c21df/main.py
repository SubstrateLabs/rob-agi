from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique subgrid pattern with the highest significance in the input grid.
    
    This function identifies subgrids that appear only once in the input grid and scores them based on
    their color complexity, unique patterns, and structural significance. It returns the subgrid with 
    the highest significance score, prioritizing patterns with clear internal structure and uniqueness.
    
    The function scans the input grid for all possible subgrids, checking if each subgrid is unique
    and calculating its significance score based on color transitions, symmetry, and pattern complexity.
    It focuses on finding patterns that are structurally significant rather than just simple or small.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique subgrid with the highest structural significance
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

    def calculate_structural_significance(subgrid: ColoredGrid) -> float:
        height, width = subgrid.get_dimensions()
        significance = 0
        
        # Prefer larger subgrids
        significance += height * width
        
        # Check for color diversity
        unique_colors = len(set(color for row in subgrid.values for color in row))
        significance += unique_colors * 2
        
        # Check for patterns
        for i in range(height):
            if len(set(subgrid.values[i])) > 1:  # Row has multiple colors
                significance += 3
        for j in range(width):
            if len(set(subgrid.values[r][j] for r in range(height))) > 1:  # Column has multiple colors
                significance += 3
        
        # Check for symmetry
        if all(subgrid.values[i] == subgrid.values[-i-1] for i in range(height//2)):
            significance += 5
        if all(subgrid.values[i][j] == subgrid.values[i][-j-1] for i in range(height) for j in range(width//2)):
            significance += 5
        
        # Check for alternating patterns
        for row in subgrid.values:
            if len(set(row[::2])) == 1 and len(set(row[1::2])) == 1 and row[0] != row[1]:
                significance += 4
        for col in zip(*subgrid.values):
            if len(set(col[::2])) == 1 and len(set(col[1::2])) == 1 and col[0] != col[1]:
                significance += 4
        
        return significance

    best_subgrid = None
    best_score = -1

    for height in range(2, min(rows, cols) + 1):
        for width in range(2, min(rows, cols) + 1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_unique_subgrid(subgrid, top, left):
                        score = calculate_structural_significance(subgrid)
                        if score > best_score:
                            best_score = score
                            best_subgrid = subgrid

    return best_subgrid
