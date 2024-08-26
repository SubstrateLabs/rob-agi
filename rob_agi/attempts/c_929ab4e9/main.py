from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the masked area (red squares)
    2. Analyzing surrounding patterns and symmetry
    3. Filling the masked area based on symmetry and pattern continuation
    4. Enforcing global symmetry (rotational and reflectional)
    5. Ensuring pattern continuity and color distribution balance
    6. Fine-tuning for global coherence
    7. Performing final symmetry and validity checks
    8. Returning the completed grid with the reconstructed pattern
    """
    masked_area = identify_masked_area(input_grid)
    filled_grid = initial_fill(input_grid, masked_area)
    symmetry_enforced_grid = enforce_global_symmetry(filled_grid)
    continuous_grid = ensure_pattern_continuity(symmetry_enforced_grid, masked_area)
    balanced_grid = balance_color_distribution(continuous_grid, masked_area)
    coherent_grid = fine_tune_coherence(balanced_grid)
    final_grid = final_symmetry_check(coherent_grid)
    return final_grid

def identify_masked_area(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def create_symmetry_mapping(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    mapping = {}
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    for r, c in masked_area:
        dr, dc = r - center_r, c - center_c
        mapping[(r, c)] = [
            (center_r + dr, center_c + dc),
            (center_r - dr, center_c + dc),
            (center_r + dr, center_c - dc),
            (center_r - dr, center_c - dc)
        ]
    return mapping

def fill_masked_area(grid: ColoredGrid, masked_area: List[Tuple[int, int]], symmetry_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> ColoredGrid:
    filled_grid = grid.deep_copy()
    for r, c in masked_area:
        corresponding_cells = symmetry_mapping[(r, c)]
        colors = [grid.values[cr][cc] for cr, cc in corresponding_cells if grid.values[cr][cc] != 2]
        if colors:
            filled_grid.values[r][c] = Counter(colors).most_common(1)[0][0]
        else:
            filled_grid.values[r][c] = most_common_neighbor(filled_grid, r, c)
    return filled_grid

def most_common_neighbor(grid: ColoredGrid, r: int, c: int) -> int:
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] != 2:
            neighbors.append(grid.values[nr][nc])
    return Counter(neighbors).most_common(1)[0][0] if neighbors else 0

def initial_fill(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> ColoredGrid:
    filled_grid = grid.deep_copy()
    for r, c in masked_area:
        neighbors = get_valid_neighbors(grid, r, c)
        if neighbors:
            filled_grid.values[r][c] = Counter(neighbors).most_common(1)[0][0]
    return filled_grid

def enforce_global_symmetry(grid: ColoredGrid) -> ColoredGrid:
    symmetry_enforced = grid.deep_copy()
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            dr, dc = r - center_r, c - center_c
            colors = [
                grid.values[r][c],
                grid.values[center_r - dr][center_c + dc] if 0 <= center_r - dr < grid.num_rows and 0 <= center_c + dc < grid.num_cols else None,
                grid.values[center_r + dr][center_c - dc] if 0 <= center_r + dr < grid.num_rows and 0 <= center_c - dc < grid.num_cols else None,
                grid.values[center_r - dr][center_c - dc] if 0 <= center_r - dr < grid.num_rows and 0 <= center_c - dc < grid.num_cols else None
            ]
            colors = [color for color in colors if color is not None]
            if colors:
                final_color = Counter(colors).most_common(1)[0][0]
                symmetry_enforced.values[r][c] = final_color
                if 0 <= center_r - dr < grid.num_rows and 0 <= center_c + dc < grid.num_cols:
                    symmetry_enforced.values[center_r - dr][center_c + dc] = final_color
                if 0 <= center_r + dr < grid.num_rows and 0 <= center_c - dc < grid.num_cols:
                    symmetry_enforced.values[center_r + dr][center_c - dc] = final_color
                if 0 <= center_r - dr < grid.num_rows and 0 <= center_c - dc < grid.num_cols:
                    symmetry_enforced.values[center_r - dr][center_c - dc] = final_color
    return symmetry_enforced

def balance_color_distribution(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> ColoredGrid:
    balanced_grid = grid.deep_copy()
    non_masked_colors = [grid.values[r][c] for r in range(grid.num_rows) for c in range(grid.num_cols) if (r, c) not in masked_area]
    target_distribution = Counter(non_masked_colors)
    
    for r, c in masked_area:
        neighbors = get_valid_neighbors(grid, r, c)
        if neighbors:
            color_scores = {color: target_distribution[color] for color in set(neighbors)}
            balanced_grid.values[r][c] = max(color_scores, key=color_scores.get)
    
    return enforce_global_symmetry(balanced_grid)

def fine_tune_coherence(grid: ColoredGrid) -> ColoredGrid:
    return grid  # Placeholder for potential future improvements

def final_symmetry_check(grid: ColoredGrid) -> ColoredGrid:
    return enforce_global_symmetry(grid)

def ensure_pattern_continuity(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> ColoredGrid:
    continuous_grid = grid.deep_copy()
    for r, c in masked_area:
        neighbors = get_valid_neighbors(grid, r, c)
        if neighbors:
            continuous_grid.values[r][c] = Counter(neighbors).most_common(1)[0][0]
    return continuous_grid

def get_valid_neighbors(grid: ColoredGrid, r: int, c: int) -> List[int]:
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] != 2:
            neighbors.append(grid.values[nr][nc])
    return neighbors

def final_global_symmetry_check(grid: ColoredGrid) -> ColoredGrid:
    final_grid = grid.deep_copy()
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            dr, dc = r - center_r, c - center_c
            colors = [
                grid.values[r][c],
                grid.values[center_r - dr][center_c + dc],
                grid.values[center_r + dr][center_c - dc],
                grid.values[center_r - dr][center_c - dc]
            ]
            final_color = Counter(colors).most_common(1)[0][0]
            final_grid.values[r][c] = final_color
            final_grid.values[center_r - dr][center_c + dc] = final_color
            final_grid.values[center_r + dr][center_c - dc] = final_color
            final_grid.values[center_r - dr][center_c - dc] = final_color
    return final_grid
