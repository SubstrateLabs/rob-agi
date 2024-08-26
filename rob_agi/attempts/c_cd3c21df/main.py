from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique subgrid pattern with the highest significance in the input grid.
    
    This function identifies subgrids that appear only once in the input grid and scores them based on
    their color complexity, unique patterns, and size. It returns the subgrid with the highest
    significance score, prioritizing complex color arrangements and unique patterns over size.
    
    The function scans the input grid for all possible subgrids, checking if each subgrid is unique
    and calculating its significance score based on color transitions, symmetry, and rarity of colors.
    It returns the unique subgrid with the highest score.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique subgrid with the highest significance score
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

    def calculate_color_complexity(subgrid: ColoredGrid) -> float:
        complexity = 0
        height, width = subgrid.get_dimensions()
        
        # Count color transitions
        for i in range(height):
            for j in range(width):
                if i > 0 and subgrid.values[i][j] != subgrid.values[i-1][j]:
                    complexity += 1
                if j > 0 and subgrid.values[i][j] != subgrid.values[i][j-1]:
                    complexity += 1
        
        # Check for symmetry
        if all(subgrid.values[i] == subgrid.values[-i-1] for i in range(height//2)):
            complexity += 5
        if all(subgrid.values[i][j] == subgrid.values[i][-j-1] for i in range(height) for j in range(width//2)):
            complexity += 5
        
        # Check for alternating patterns
        for row in subgrid.values:
            if len(set(row[::2])) == 1 and len(set(row[1::2])) == 1 and row[0] != row[1]:
                complexity += 3
        for col in zip(*subgrid.values):
            if len(set(col[::2])) == 1 and len(set(col[1::2])) == 1 and col[0] != col[1]:
                complexity += 3
        
        # Consider color rarity
        color_counts = {}
        for row in subgrid.values:
            for color in row:
                color_counts[color] = color_counts.get(color, 0) + 1
        total_colors = sum(color_counts.values())
        for count in color_counts.values():
            rarity = 1 - (count / total_colors)
            complexity += rarity * 2
        
        return complexity

    def calculate_significance(subgrid: ColoredGrid) -> float:
        size = subgrid.get_dimensions()[0] * subgrid.get_dimensions()[1]
        unique_colors = len(set(color for row in subgrid.values for color in row))
        color_complexity = calculate_color_complexity(subgrid)
        
        return (size * 0.2) + (unique_colors * 0.3) + (color_complexity * 0.5)

    best_subgrid = None
    best_score = -1

    for height in range(2, min(rows, cols) + 1):
        for width in range(2, min(rows, cols) + 1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_unique_subgrid(subgrid, top, left):
                        score = calculate_significance(subgrid)
                        if score > best_score:
                            best_score = score
                            best_subgrid = subgrid

    return best_subgrid
