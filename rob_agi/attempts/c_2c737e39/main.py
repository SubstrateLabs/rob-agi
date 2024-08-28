from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_2c737e39(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying the main pattern,
    creating a shifted duplicate, and removing isolated gray squares.

    1. Identify the main pattern using flood-fill from the top-left.
    2. Determine the bounding box of the main pattern.
    3. Analyze available space and determine the best direction for duplication.
    4. Create a duplicate of the pattern in the chosen direction.
    5. Identify and remove isolated gray squares.
    6. Adjust the duplicate pattern if it extends beyond grid boundaries.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid with the duplicated pattern.
    """
    def flood_fill(grid: List[List[int]], start: Tuple[int, int], visited: Set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int, int]]:
        rows, cols = len(grid), len(grid[0])
        color = grid[start[0]][start[1]]
        if color == 0:
            return [], (0, 0, 0, 0)
        
        pattern = []
        stack = [start]
        min_r, min_c, max_r, max_c = rows, cols, 0, 0

        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid[r][c] != 0:
                visited.add((r, c))
                pattern.append((r, c, grid[r][c]))
                min_r, min_c = min(min_r, r), min(min_c, c)
                max_r, max_c = max(max_r, r), max(max_c, c)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        
        return pattern, (min_r, min_c, max_r, max_c)

    def is_isolated_gray(grid: List[List[int]], r: int, c: int) -> bool:
        rows, cols = len(grid), len(grid[0])
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0:
                return False
        return True

    input_values = input_grid.values
    pattern, (min_r, min_c, max_r, max_c) = flood_fill(input_values, (0, 0), set())
    
    pattern_height = max_r - min_r + 1
    pattern_width = max_c - min_c + 1
    
    output_grid = input_grid.deep_copy()
    output_values = output_grid.values
    rows, cols = len(output_values), len(output_values[0])

    # Determine duplication direction
    space_down = rows - max_r - 1
    space_right = cols - max_c - 1
    space_left = min_c
    space_up = min_r

    if space_down >= pattern_height:
        shift_r, shift_c = pattern_height, 0
    elif space_right >= pattern_width:
        shift_r, shift_c = 0, pattern_width
    elif space_left >= pattern_width:
        shift_r, shift_c = 0, -pattern_width
    elif space_up >= pattern_height:
        shift_r, shift_c = -pattern_height, 0
    else:
        # If no space available, don't duplicate
        shift_r, shift_c = 0, 0

    # Create duplicate pattern
    for r, c, color in pattern:
        if color != 5:  # Don't duplicate gray squares
            new_r, new_c = r + shift_r, c + shift_c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                output_values[new_r][new_c] = color

    # Remove isolated gray squares
    for r in range(rows):
        for c in range(cols):
            if output_values[r][c] == 5 and is_isolated_gray(output_values, r, c):
                output_values[r][c] = 0

    return output_grid
