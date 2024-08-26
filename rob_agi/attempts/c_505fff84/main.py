from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import numpy as np

def solve_505fff84(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the most significant pattern of red squares from the input grid.
    
    The function performs the following steps:
    1. Creates a heat map based on the density of red squares
    2. Identifies the most representative area using the heat map
    3. Extracts and simplifies the selected area
    4. Refines the pattern by removing unnecessary rows/columns
    5. Ensures pattern integrity and minimum size
    6. Returns the final pattern as a new ColoredGrid
    """
    # Step 1: Create heat map
    heat_map = create_heat_map(input_grid)
    
    # Step 2: Identify the most representative area
    top, left, bottom, right = find_best_area(heat_map)
    
    # Step 3: Extract and simplify the selected area
    subgrid = input_grid.extract_subgrid(top, left, bottom - top + 1, right - left + 1)
    simplified = simplify_subgrid(subgrid)
    
    # Step 4: Refine the pattern
    refined = refine_pattern(simplified)
    
    # Step 5: Ensure pattern integrity and minimum size
    final_pattern = ensure_pattern_integrity(refined, subgrid)
    
    return final_pattern

def create_heat_map(grid: ColoredGrid) -> np.ndarray:
    values = np.array(grid.values)
    kernel = np.ones((3, 3))
    heat_map = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            subgrid = values[max(0, i-1):min(i+2, values.shape[0]), max(0, j-1):min(j+2, values.shape[1])]
            heat_map[i, j] = np.sum(subgrid == 2)
    return heat_map

def find_best_area(heat_map: np.ndarray) -> Tuple[int, int, int, int]:
    best_score = -1
    best_area = (0, 0, heat_map.shape[0] - 1, heat_map.shape[1] - 1)
    for top in range(heat_map.shape[0]):
        for left in range(heat_map.shape[1]):
            for bottom in range(top + 2, heat_map.shape[0]):
                for right in range(left + 2, heat_map.shape[1]):
                    area = heat_map[top:bottom+1, left:right+1]
                    score = np.mean(area)
                    if score > best_score:
                        best_score = score
                        best_area = (top, left, bottom, right)
    return best_area

def simplify_subgrid(grid: ColoredGrid) -> ColoredGrid:
    new_values = [[2 if cell == 2 else 0 for cell in row] for row in grid.values]
    return ColoredGrid(values=new_values)

def refine_pattern(grid: ColoredGrid) -> ColoredGrid:
    values = np.array(grid.values)
    row_ratios = np.mean(values == 2, axis=1)
    col_ratios = np.mean(values == 2, axis=0)
    rows_to_keep = np.where(row_ratios >= 0.3)[0]
    cols_to_keep = np.where(col_ratios >= 0.3)[0]
    refined_values = values[rows_to_keep][:, cols_to_keep].tolist()
    return ColoredGrid(values=refined_values)

def ensure_pattern_integrity(grid: ColoredGrid, original_subgrid: ColoredGrid) -> ColoredGrid:
    values = np.array(grid.values)
    if values.shape[0] < 2 or values.shape[1] < 2:
        original_values = np.array(original_subgrid.values)
        if values.shape[0] < 2:
            values = np.vstack((values, original_values[values.shape[0]]))
        if values.shape[1] < 2:
            values = np.hstack((values, original_values[:, values.shape[1]].reshape(-1, 1)))
    
    # Ensure at least one red square
    if np.sum(values == 2) == 0:
        red_positions = np.argwhere(np.array(original_subgrid.values) == 2)
        if len(red_positions) > 0:
            values[tuple(red_positions[0])] = 2
    
    return ColoredGrid(values=values.tolist())
