from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque, Counter

def solve_cfb2ce5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding each color while preserving the original pattern
    and maintaining a black border.

    The algorithm works as follows:
    1. Analyze the initial grid to identify unique colors and their patterns.
    2. Perform a staged expansion process:
       a. Pattern-based expansion: Expand each color based on its identified pattern.
       b. Directional expansion: Expand colors in main directions based on current configuration.
       c. Space filling: Fill remaining empty cells to create largest contiguous areas.
    3. Maintain the black border throughout the process.
    4. Iterate the expansion process until no more changes can be made.
    5. Perform a final check to ensure all non-border cells are filled and patterns are maintained.
    6. Return the transformed grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion algorithm.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_unique_colors() -> List[int]:
        return sorted(set(grid.values[r][c] for r in range(rows) for c in range(cols) if grid.values[r][c] != 0))

    def get_color_positions(color: int) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == color]

    def identify_pattern(color: int) -> str:
        positions = get_color_positions(color)
        if len(positions) < 2:
            return "isolated"
        
        diffs = [(positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1]) for i in range(len(positions)-1)]
        if all(diff == (0, 1) or diff == (1, 0) for diff in diffs):
            return "line"
        elif all((abs(diff[0]), abs(diff[1])) == (1, 1) for diff in diffs):
            return "diagonal"
        elif len(set(diffs)) > 1:
            return "complex"
        else:
            return "cluster"

    def expand_pattern(color: int, pattern: str):
        positions = get_color_positions(color)
        new_positions = []
        if pattern == "line":
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        elif pattern == "diagonal":
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for r, c in positions:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = color
                    new_positions.append((nr, nc))
        return new_positions

    def directional_expansion(color: int):
        positions = get_color_positions(color)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        new_positions = []
        for r, c in positions:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = color
                    new_positions.append((nr, nc))
        return new_positions

    def fill_space():
        changed = False
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid.values[r][c] == 0:
                    neighbors = [grid.values[r+dr][c+dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]]
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    if non_zero_neighbors:
                        most_common = Counter(non_zero_neighbors).most_common(1)[0][0]
                        grid.values[r][c] = most_common
                        changed = True
        return changed

    def maintain_border():
        for r in range(rows):
            grid.values[r][0] = grid.values[r][-1] = 0
        for c in range(cols):
            grid.values[0][c] = grid.values[-1][c] = 0

    colors = get_unique_colors()
    color_patterns = {color: identify_pattern(color) for color in colors}
    
    changed = True
    while changed:
        changed = False
        for color in colors:
            new_positions = expand_pattern(color, color_patterns[color])
            if new_positions:
                changed = True
        
        if not changed:
            for color in colors:
                new_positions = directional_expansion(color)
                if new_positions:
                    changed = True
        
        if not changed:
            changed = fill_space()
        
        maintain_border()

    return grid
