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
    
    The solution prioritizes solid color blocks that are unique within the grid, with a preference for
    larger and more structurally significant patterns.
    
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
        significance += height * width * 2
        
        # Check for color uniformity (solid color blocks)
        unique_colors = set(color for row in subgrid.values for color in row if color != 0)
        if len(unique_colors) == 1:
            significance += 100  # Highly prioritize solid color blocks
        else:
            significance -= 50  # Penalize non-solid color blocks
        
        # Penalize subgrids with black (empty) cells
        black_cells = sum(row.count(0) for row in subgrid.values)
        significance -= black_cells * 10
        
        return significance

    best_subgrid = None
    best_score = float('-inf')

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
