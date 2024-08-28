from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue regions of size 2-9 pixels with maximum dimension <= 3 are changed to red or green.
    2. Blue plus sign shapes (5 pixels in a cross pattern) are also changed.
    3. The target color (red or green) is determined by the global prevalence of these colors.
    4. Larger blue regions, single blue pixels, and other colors remain unchanged.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_connected_region(row: int, col: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        region = []
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and output_grid.values[r][c] == color and (r, c) not in visited:
                region.append((r, c))
                visited.add((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
        return region

    def should_transform_region(region: List[Tuple[int, int]]) -> bool:
        if len(region) == 1 or len(region) > 9:
            return False
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        
        # Check for plus sign shape
        if len(region) == 5:
            center = (min_row + max_row) // 2, (min_col + max_col) // 2
            plus_shape = {center, (center[0]-1, center[1]), (center[0]+1, center[1]), 
                          (center[0], center[1]-1), (center[0], center[1]+1)}
            if set(region) == plus_shape:
                return True
        
        return len(region) <= 9 and max(height, width) <= 3

    # Determine global color prevalence
    red_count = sum(row.count(2) for row in output_grid.values)
    green_count = sum(row.count(3) for row in output_grid.values)
    preferred_color = 2 if red_count >= green_count else 3

    visited = set()
    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1 and (row, col) not in visited:
                region = find_connected_region(row, col, 1, visited)
                if should_transform_region(region):
                    for r, c in region:
                        output_grid.values[r][c] = preferred_color

    return output_grid
