from rob_agi.colored_grid import ColoredGrid
from collections import Counter, deque
from typing import List, Tuple, Dict, Set

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Identify all black regions in the grid.
    2. For each black region, analyze the surrounding non-black area to detect patterns.
    3. Extend detected patterns into the black regions.
    4. Apply smoothing and consistency checks to ensure visual coherence.
    5. Handle edge cases and perform final refinements.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_neighbors(r: int, c: int, include_diagonal: bool = False) -> List[Tuple[int, int, int]]:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return [(r + dr, c + dc, grid.get_cell(r + dr, c + dc)) 
                for dr, dc in directions 
                if 0 <= r + dr < rows and 0 <= c + dc < cols]

    def find_black_regions() -> List[Set[Tuple[int, int]]]:
        visited = set()
        black_regions = []
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0 and (r, c) not in visited:
                    region = set()
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 0:
                            visited.add((curr_r, curr_c))
                            region.add((curr_r, curr_c))
                            queue.extend(get_neighbors(curr_r, curr_c))
                    black_regions.append(region)
        return black_regions

    def analyze_surrounding(region: Set[Tuple[int, int]]) -> Dict[int, int]:
        surrounding_colors = Counter()
        for r, c in region:
            for nr, nc, color in get_neighbors(r, c, include_diagonal=True):
                if color != 0:
                    surrounding_colors[color] += 1
        return surrounding_colors

    def detect_pattern(surrounding_colors: Dict[int, int]) -> List[int]:
        total = sum(surrounding_colors.values())
        pattern = []
        for color, count in surrounding_colors.items():
            pattern.extend([color] * (count * 10 // total))  # Normalize to a scale of 10
        return pattern

    def fill_region(region: Set[Tuple[int, int]], pattern: List[int]):
        pattern_index = 0
        for r, c in region:
            grid.set_cell(r, c, pattern[pattern_index])
            pattern_index = (pattern_index + 1) % len(pattern)

    def smooth_transitions():
        for r in range(rows):
            for c in range(cols):
                neighbors = get_neighbors(r, c, include_diagonal=True)
                color_counts = Counter(color for _, _, color in neighbors if color != 0)
                if color_counts:
                    most_common_color = color_counts.most_common(1)[0][0]
                    grid.set_cell(r, c, most_common_color)

    # Main algorithm steps
    black_regions = find_black_regions()
    for region in black_regions:
        surrounding_colors = analyze_surrounding(region)
        pattern = detect_pattern(surrounding_colors)
        fill_region(region, pattern)

    # Apply smoothing
    smooth_transitions()

    return grid
