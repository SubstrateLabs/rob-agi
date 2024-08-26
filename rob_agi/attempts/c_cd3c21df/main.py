from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the unique, contiguous pattern with the highest structural significance in the input grid.
    
    This function scans the input grid for all possible subgrids, identifying unique contiguous patterns.
    It prioritizes larger patterns and those that are not black (empty space). The function returns the
    pattern that is both unique within the grid and has the highest structural significance.
    
    The structural significance is determined by the size of the pattern, its colors, and its uniqueness.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The unique contiguous pattern with the highest structural significance
    """
    rows, cols = input_grid.get_dimensions()
    
    def is_contiguous(subgrid: ColoredGrid) -> bool:
        height, width = subgrid.get_dimensions()
        visited = set()
        stack = [(0, 0)]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and subgrid.values[nr][nc] != 0:
                        stack.append((nr, nc))
        return len(visited) == sum(1 for row in subgrid.values for cell in row if cell != 0)
    
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
        non_zero_cells = sum(1 for row in subgrid.values for cell in row if cell != 0)
        unique_colors = len(set(cell for row in subgrid.values for cell in row if cell != 0))
        return height * width * non_zero_cells * unique_colors

    best_subgrid = None
    best_score = float('-inf')

    for height in range(rows, 0, -1):
        for width in range(cols, 0, -1):
            for top in range(rows - height + 1):
                for left in range(cols - width + 1):
                    subgrid = input_grid.extract_subgrid(top, left, height, width)
                    if is_contiguous(subgrid) and is_unique_subgrid(subgrid, top, left):
                        score = calculate_significance(subgrid)
                        if score > best_score:
                            best_score = score
                            best_subgrid = subgrid

    return best_subgrid
