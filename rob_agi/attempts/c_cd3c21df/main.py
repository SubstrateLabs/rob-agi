from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique, largest subgrid pattern with the highest significance in the input grid.
    
    This function identifies subgrids that appear only once in the input grid and scores them based on
    their size, number of unique colors, and specific color arrangements. It returns the subgrid with
    the highest significance score, prioritizing larger and more complex patterns.
    
    The function scans the input grid from largest possible subgrid to smallest, checking if each
    subgrid is unique and calculating its significance score. It returns the unique subgrid with the
    highest score.
    
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

    def calculate_significance(subgrid: ColoredGrid) -> float:
        height, width = subgrid.get_dimensions()
        unique_colors: Set[int] = set()
        score = height * width  # Base score is the area

        for row in subgrid.values:
            unique_colors.update(row)
        
        score *= len(unique_colors)  # Multiply by number of unique colors

        # Bonus for alternating patterns
        for row in subgrid.values:
            if len(set(row[::2])) == 1 and len(set(row[1::2])) == 1 and row[0] != row[1]:
                score += width
        for col in zip(*subgrid.values):
            if len(set(col[::2])) == 1 and len(set(col[1::2])) == 1 and col[0] != col[1]:
                score += height

        # Bonus for border patterns
        if len(set(subgrid.values[0] + subgrid.values[-1] + [row[0] for row in subgrid.values] + [row[-1] for row in subgrid.values])) > 1:
            score += 2 * (height + width)

        return score

    best_subgrid = None
    best_score = -1

    for height in range(min(rows, cols), 1, -1):  # Start from smaller dimension
        for width in range(min(rows, cols), 1, -1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_unique_subgrid(subgrid, top, left):
                        score = calculate_significance(subgrid)
                        if score > best_score or (score == best_score and (top, left) < (best_subgrid._top, best_subgrid._left)):
                            best_score = score
                            best_subgrid = subgrid
                            best_subgrid._top, best_subgrid._left = top, left  # Store position for tiebreaking

    return best_subgrid
