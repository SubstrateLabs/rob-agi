from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

def solve_929ab4e9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by:
    1. Identifying the masked area (red squares)
    2. Analyzing global patterns and symmetry
    3. Creating a symmetry mapping for masked cells
    4. Performing initial fill based on neighbors and symmetry
    5. Enforcing global symmetry (rotational and reflectional)
    6. Ensuring pattern continuity
    7. Balancing color distribution
    8. Performing final symmetry and coherence checks
    9. Returning the completed grid with the reconstructed pattern

    The solution maintains global symmetry, continues patterns from non-masked areas,
    adapts the filling strategy based on specific grid characteristics, preserves
    consistency in color distribution, and takes into account the broader context
    of the entire grid. It handles various grid patterns and masked area configurations
    by applying a step-by-step approach that considers both local and global features.
    """
    masked_area = identify_masked_area(input_grid)
    symmetry_mapping = create_symmetry_mapping(input_grid, masked_area)
    filled_grid = initial_fill(input_grid, masked_area, symmetry_mapping)
    symmetry_enforced_grid = enforce_global_symmetry(filled_grid)
    continuous_grid = ensure_pattern_continuity(symmetry_enforced_grid, masked_area)
    balanced_grid = balance_color_distribution(continuous_grid, masked_area)
    final_grid = final_global_symmetry_check(balanced_grid)
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

def initial_fill(grid: ColoredGrid, masked_area: List[Tuple[int, int]], symmetry_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> ColoredGrid:
    filled_grid = grid.deep_copy()
    for r, c in masked_area:
        neighbors = get_valid_neighbors(grid, r, c)
        symmetric_colors = [grid.values[sr][sc] for sr, sc in symmetry_mapping[(r, c)] if grid.values[sr][sc] != 2]
        all_colors = neighbors + symmetric_colors
        if all_colors:
            filled_grid.values[r][c] = Counter(all_colors).most_common(1)[0][0]
        else:
            filled_grid.values[r][c] = most_common_color(grid)
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

def ensure_pattern_continuity(grid: ColoredGrid, masked_area: List[Tuple[int, int]]) -> ColoredGrid:
    continuous_grid = grid.deep_copy()
    for r, c in masked_area:
        neighbors = get_valid_neighbors(grid, r, c)
        if neighbors:
            pattern = detect_pattern(grid, r, c)
            if pattern:
                continuous_grid.values[r][c] = pattern[0]
            else:
                continuous_grid.values[r][c] = Counter(neighbors).most_common(1)[0][0]
    return continuous_grid

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

def final_global_symmetry_check(grid: ColoredGrid) -> ColoredGrid:
    final_grid = grid.deep_copy()
    center_r, center_c = grid.num_rows // 2, grid.num_cols // 2
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            dr, dc = r - center_r, c - center_c
            colors = [grid.values[r][c]]
            symmetric_positions = [
                (center_r - dr, center_c + dc),
                (center_r + dr, center_c - dc),
                (center_r - dr, center_c - dc)
            ]
            for sr, sc in symmetric_positions:
                if 0 <= sr < grid.num_rows and 0 <= sc < grid.num_cols:
                    colors.append(grid.values[sr][sc])
            
            final_color = Counter(colors).most_common(1)[0][0]
            final_grid.values[r][c] = final_color
            for sr, sc in symmetric_positions:
                if 0 <= sr < grid.num_rows and 0 <= sc < grid.num_cols:
                    final_grid.values[sr][sc] = final_color
    return final_grid

def get_valid_neighbors(grid: ColoredGrid, r: int, c: int) -> List[int]:
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] != 2:
            neighbors.append(grid.values[nr][nc])
    return neighbors

def most_common_color(grid: ColoredGrid) -> int:
    all_colors = [grid.values[r][c] for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] != 2]
    return Counter(all_colors).most_common(1)[0][0]

def detect_pattern(grid: ColoredGrid, r: int, c: int) -> List[int]:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        pattern = []
        for i in range(1, 4):  # Check up to 3 cells in each direction
            nr, nc = r + i*dr, c + i*dc
            if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.values[nr][nc] != 2:
                pattern.append(grid.values[nr][nc])
            else:
                break
        if len(pattern) >= 2 and len(set(pattern)) > 1:  # At least 2 different colors
            return pattern
    return []
