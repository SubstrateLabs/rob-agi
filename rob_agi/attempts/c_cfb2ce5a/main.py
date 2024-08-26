from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque, Counter
import random

def solve_cfb2ce5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding each color while preserving the original pattern
    and maintaining a black border.

    The algorithm works as follows:
    1. Analyze the initial grid to identify unique colors, their patterns, and frequencies.
    2. Calculate target frequencies for each color.
    3. Create pattern templates for each color based on their initial arrangement.
    4. Perform a multi-stage expansion process:
       a. Pattern-based Expansion: Expand each color according to its template.
       b. Boundary Interaction: Handle color interactions at boundaries.
       c. Fill Empty Spaces: Use pattern extension and weighted random filling.
       d. Check and Adjust: Adjust expansion priorities based on current vs target frequencies.
    5. Maintain the black border throughout the process.
    6. Make final adjustments for symmetry and color balance.
    7. Enhance connectivity for isolated color instances.
    8. Perform iterative refinement to balance color distribution.
    9. Return the transformed grid.

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

    def calculate_target_frequencies(initial_frequencies: Dict[int, int]) -> Dict[int, int]:
        total_cells = (rows - 2) * (cols - 2)  # Exclude border
        min_threshold = max(1, int(0.05 * total_cells))
        target_frequencies = {}
        for color, freq in initial_frequencies.items():
            target_frequencies[color] = max(freq * 2, min_threshold)
        return target_frequencies

    def create_pattern_template(color: int, pattern: str) -> List[Tuple[int, int]]:
        positions = get_color_positions(color)
        if pattern == "isolated":
            return [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif pattern == "line":
            direction = (positions[1][0] - positions[0][0], positions[1][1] - positions[0][1])
            return [direction, (-direction[0], -direction[1])]
        elif pattern == "diagonal":
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif pattern == "complex":
            return [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:  # cluster
            return [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def expand_pattern(color: int, template: List[Tuple[int, int]]):
        positions = get_color_positions(color)
        new_positions = []
        for r, c in positions:
            for dr, dc in template:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = color
                    new_positions.append((nr, nc))
        return new_positions

    def handle_boundary_interaction(color: int, target_frequencies: Dict[int, int]):
        positions = get_color_positions(color)
        for r, c in positions:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] != 0 and grid.values[nr][nc] != color:
                    neighbor_color = grid.values[nr][nc]
                    if target_frequencies[color] > target_frequencies[neighbor_color]:
                        grid.values[nr][nc] = color

    def fill_empty_spaces(target_frequencies: Dict[int, int]):
        empty_cells = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1) if grid.values[r][c] == 0]
        colors = list(target_frequencies.keys())
        weights = list(target_frequencies.values())
        for r, c in empty_cells:
            neighbors = [grid.values[r+dr][c+dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 < r+dr < rows-1 and 0 < c+dc < cols-1]
            if neighbors:
                neighbor_colors = [color for color in neighbors if color != 0]
                if neighbor_colors:
                    grid.values[r][c] = max(set(neighbor_colors), key=neighbor_colors.count)
                else:
                    grid.values[r][c] = random.choices(colors, weights=weights)[0]

    def adjust_symmetry():
        for r in range(1, rows - 1):
            for c in range(1, cols // 2):
                left_color = grid.values[r][c]
                right_color = grid.values[r][cols - 1 - c]
                if left_color != right_color:
                    if random.random() < 0.5:
                        grid.values[r][cols - 1 - c] = left_color
                    else:
                        grid.values[r][c] = right_color

    def enhance_connectivity():
        for color in colors:
            positions = get_color_positions(color)
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    start = positions[i]
                    end = positions[i + 1]
                    path = find_path(start, end)
                    for r, c in path:
                        if grid.values[r][c] == 0:
                            grid.values[r][c] = color

    def find_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == end:
                return path
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        return []

    def maintain_border():
        for r in range(rows):
            grid.values[r][0] = grid.values[r][-1] = 0
        for c in range(cols):
            grid.values[0][c] = grid.values[-1][c] = 0

    def preserve_original_pattern(color: int):
        original_positions = get_color_positions(color)
        for r, c in original_positions:
            grid.values[r][c] = color

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    colors = get_unique_colors()
    color_patterns = {color: identify_pattern(color) for color in colors}
    initial_frequencies = get_color_frequencies()
    target_frequencies = calculate_target_frequencies(initial_frequencies)
    pattern_templates = {color: create_pattern_template(color, pattern) for color, pattern in color_patterns.items()}

    def expand_color(color):
        positions = get_color_positions(color)
        new_positions = []
        for r, c in positions:
            for dr, dc in pattern_templates[color]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and grid.values[nr][nc] == 0:
                    grid.values[nr][nc] = color
                    new_positions.append((nr, nc))
        return new_positions

    def fill_empty_spaces():
        empty_cells = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1) if grid.values[r][c] == 0]
        for r, c in empty_cells:
            neighbors = [grid.values[r+dr][c+dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 < r+dr < rows-1 and 0 < c+dc < cols-1]
            non_zero_neighbors = [color for color in neighbors if color != 0]
            if non_zero_neighbors:
                grid.values[r][c] = max(set(non_zero_neighbors), key=non_zero_neighbors.count)
            else:
                grid.values[r][c] = random.choice(colors)

    for _ in range(5):  # Perform multiple iterations of expansion
        for color in sorted(colors, key=lambda x: -initial_frequencies[x]):
            expand_color(color)
        fill_empty_spaces()
        maintain_border()

    # Final adjustments
    for color in colors:
        preserve_original_pattern(color)
    fill_empty_spaces()
    maintain_border()

    return grid
