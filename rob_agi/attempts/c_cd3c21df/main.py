from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict, Set

def solve_cd3c21df(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Find the most significant unique pattern in the input grid.
    
    This function identifies contiguous regions of non-black cells,
    evaluates their significance based on size, color diversity,
    compactness, and uniqueness, and returns the most significant
    unique pattern.
    
    Args:
    input_grid (ColoredGrid): The input grid to analyze

    Returns:
    ColoredGrid: The most significant unique pattern
    """
    rows, cols = input_grid.get_dimensions()
    
    def find_contiguous_regions():
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and input_grid.values[r][c] != 0:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) not in visited:
                            visited.add((cr, cc))
                            region.append((cr, cc))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 0:
                                    stack.append((nr, nc))
                    regions.append(region)
        return regions

    def extract_pattern(region):
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        pattern = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(input_grid.values[r][c] if (r, c) in region else 0)
            pattern.append(row)
        return ColoredGrid(values=pattern)

    def is_unique(pattern):
        pattern_height, pattern_width = pattern.get_dimensions()
        for r in range(rows - pattern_height + 1):
            for c in range(cols - pattern_width + 1):
                if all(input_grid.values[r+i][c+j] == pattern.values[i][j]
                       for i in range(pattern_height)
                       for j in range(pattern_width)):
                    return False
        return True

    def calculate_significance(pattern):
        height, width = pattern.get_dimensions()
        non_zero_cells = sum(1 for row in pattern.values for cell in row if cell != 0)
        unique_colors = len(set(cell for row in pattern.values for cell in row if cell != 0))
        compactness = non_zero_cells / (height * width)
        return non_zero_cells * unique_colors * compactness

    regions = find_contiguous_regions()
    best_pattern = None
    best_score = float('-inf')

    for region in regions:
        pattern = extract_pattern(region)
        if is_unique(pattern):
            score = calculate_significance(pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern

    return best_pattern if best_pattern else ColoredGrid(values=[[0]])
