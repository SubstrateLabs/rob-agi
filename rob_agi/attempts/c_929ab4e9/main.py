from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the masked area (red squares)
    2. Analyzing the global pattern and symmetry
    3. Creating a pattern prediction model
    4. Filling the masked area based on the global pattern
    5. Enforcing symmetry and pattern continuity
    6. Performing a final global pattern check
    7. Returning the completed grid with the reconstructed pattern
    """
    masked_area = identify_masked_area(input_grid)
    global_pattern = analyze_global_pattern(input_grid, masked_area)
    symmetry_axes = find_symmetry_axes(input_grid)
    prediction_model = create_pattern_prediction_model(input_grid, global_pattern, symmetry_axes)
    filled_grid = fill_masked_area(input_grid, masked_area, prediction_model)
    symmetry_enforced_grid = enforce_symmetry(filled_grid, symmetry_axes)
    pattern_continuous_grid = ensure_pattern_continuity(symmetry_enforced_grid, global_pattern)
    return final_global_pattern_check(pattern_continuous_grid, global_pattern)

def identify_masked_area(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def analyze_global_pattern(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> Dict:
    pattern = {}
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in masked_area:
                color = grid.values[r][c]
                if (r % 2, c % 2) not in pattern:
                    pattern[(r % 2, c % 2)] = []
                pattern[(r % 2, c % 2)].append(color)
    return {k: max(set(v), key=v.count) for k, v in pattern.items()}

def find_symmetry_axes(grid: ColoredGrid) -> Tuple[bool, bool, bool, bool]:
    horizontal = all(grid.values[r] == grid.values[-r-1] for r in range(grid.num_rows // 2))
    vertical = all(row[c] == row[-c-1] for row in grid.values for c in range(grid.num_cols // 2))
    diagonal1 = all(grid.values[r][c] == grid.values[c][r] for r in range(grid.num_rows) for c in range(grid.num_cols))
    diagonal2 = all(grid.values[r][c] == grid.values[grid.num_rows-1-c][grid.num_cols-1-r] 
                    for r in range(grid.num_rows) for c in range(grid.num_cols))
    return horizontal, vertical, diagonal1, diagonal2

def create_pattern_prediction_model(grid: ColoredGrid, global_pattern: Dict, symmetry_axes: Tuple[bool, bool, bool, bool]) -> callable:
    def predict_color(r: int, c: int) -> int:
        candidates = [global_pattern.get((r % 2, c % 2), 0)]
        horizontal, vertical, diagonal1, diagonal2 = symmetry_axes
        if horizontal:
            candidates.append(grid.values[grid.num_rows-1-r][c])
        if vertical:
            candidates.append(grid.values[r][grid.num_cols-1-c])
        if diagonal1:
            candidates.append(grid.values[c][r])
        if diagonal2:
            candidates.append(grid.values[grid.num_rows-1-c][grid.num_cols-1-r])
        return max(set(candidates), key=candidates.count)
    return predict_color

def fill_masked_area(grid: ColoredGrid, masked_area: List[Tuple[int, int]], prediction_model: callable) -> ColoredGrid:
    filled_grid = grid.deep_copy()
    for r, c in masked_area:
        filled_grid.values[r][c] = prediction_model(r, c)
    return filled_grid

def enforce_symmetry(grid: ColoredGrid, symmetry_axes: Tuple[bool, bool, bool, bool]) -> ColoredGrid:
    horizontal, vertical, diagonal1, diagonal2 = symmetry_axes
    symmetry_enforced = grid.deep_copy()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            colors = [symmetry_enforced.values[r][c]]
            if horizontal:
                colors.append(symmetry_enforced.values[grid.num_rows-1-r][c])
            if vertical:
                colors.append(symmetry_enforced.values[r][grid.num_cols-1-c])
            if diagonal1:
                colors.append(symmetry_enforced.values[c][r])
            if diagonal2:
                colors.append(symmetry_enforced.values[grid.num_rows-1-c][grid.num_cols-1-r])
            final_color = max(set(colors), key=colors.count)
            symmetry_enforced.values[r][c] = final_color
            if horizontal:
                symmetry_enforced.values[grid.num_rows-1-r][c] = final_color
            if vertical:
                symmetry_enforced.values[r][grid.num_cols-1-c] = final_color
            if diagonal1:
                symmetry_enforced.values[c][r] = final_color
            if diagonal2:
                symmetry_enforced.values[grid.num_rows-1-c][grid.num_cols-1-r] = final_color
    return symmetry_enforced

def ensure_pattern_continuity(grid: ColoredGrid, global_pattern: Dict) -> ColoredGrid:
    continuous_grid = grid.deep_copy()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            expected_color = global_pattern.get((r % 2, c % 2), continuous_grid.values[r][c])
            if continuous_grid.values[r][c] != expected_color:
                continuous_grid.values[r][c] = expected_color
    return continuous_grid

def final_global_pattern_check(grid: ColoredGrid, global_pattern: Dict) -> ColoredGrid:
    final_grid = grid.deep_copy()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            expected_color = global_pattern.get((r % 2, c % 2), final_grid.values[r][c])
            if final_grid.values[r][c] != expected_color:
                final_grid.values[r][c] = expected_color
    return final_grid
