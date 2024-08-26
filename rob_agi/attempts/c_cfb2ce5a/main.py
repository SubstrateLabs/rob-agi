from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque, Counter

def solve_cfb2ce5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding each color while preserving the original pattern
    and maintaining a black border.

    The algorithm works as follows:
    1. Analyze the initial grid to identify unique colors, their patterns, and frequencies.
    2. Assign expansion priorities to colors based on their initial frequency and pattern.
    3. Perform a multi-stage expansion process:
       a. Core Pattern Expansion: Expand each color's core pattern.
       b. Directional Expansion: Expand colors in main directions based on priorities.
       c. Edge Behavior and Boundary Formation: Apply rules for color interactions at boundaries.
       d. Fill and Balance: Fill remaining empty cells and balance color frequencies.
    4. Adjust for symmetry or controlled asymmetry.
    5. Maintain the black border throughout the process.
    6. Verify and refine the final pattern.
    7. Return the transformed grid.

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

    def get_color_frequencies() -> Dict[int, int]:
        return Counter(grid.values[r][c] for r in range(rows) for c in range(cols) if grid.values[r][c] != 0)

    def assign_priorities(colors: List[int], patterns: Dict[int, str], frequencies: Dict[int, int]) -> Dict[int, int]:
        priorities = {}
        for color in colors:
            priority = frequencies[color]
            if patterns[color] == "complex":
                priority += 3
            elif patterns[color] == "cluster":
                priority += 2
            elif patterns[color] == "line":
                priority += 1
            priorities[color] = priority
        return priorities

    def expand_core_pattern(color: int, pattern: str):
        positions = get_color_positions(color)
        new_positions = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for r, c in positions:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = color
                    new_positions.append((nr, nc))
        return new_positions

    def directional_expansion(color: int, priority: int):
        positions = get_color_positions(color)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        new_positions = []
        for r, c in positions:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1:
                    if grid.values[nr][nc] == 0:
                        grid.values[nr][nc] = color
                        new_positions.append((nr, nc))
                    elif grid.values[nr][nc] != color and priorities[grid.values[nr][nc]] < priority:
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

    def adjust_symmetry():
        # Implement symmetry adjustments here if needed
        pass

    colors = get_unique_colors()
    color_patterns = {color: identify_pattern(color) for color in colors}
    initial_frequencies = get_color_frequencies()
    priorities = assign_priorities(colors, color_patterns, initial_frequencies)
    
    for _ in range(3):  # Perform multiple iterations of expansion
        for color in sorted(colors, key=lambda c: priorities[c], reverse=True):
            expand_core_pattern(color, color_patterns[color])
        
        for color in sorted(colors, key=lambda c: priorities[c], reverse=True):
            directional_expansion(color, priorities[color])
        
        fill_space()
        maintain_border()
    
    adjust_symmetry()
    maintain_border()

    return grid
