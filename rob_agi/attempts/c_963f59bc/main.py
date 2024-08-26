from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
import copy

def solve_963f59bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the primary shape and replicating it
    for each isolated square of a different color. The replicated shapes are
    rotated, mirrored, and positioned to maximize coverage while adhering to constraints.
    
    1. Identifies the primary shape (largest non-black connected region)
    2. Finds isolated squares of different colors
    3. For each isolated square, applies a transformed version of the primary shape:
       - Generates all possible transformations (rotations and mirrors)
       - Tries various placements around the isolated square
       - Selects the placement that maximizes new colored squares without overlapping
    4. Applies transformations for all isolated squares
    5. Preserves original elements and grid boundaries
    6. Returns the modified grid with all valid transformations applied
    """
    def find_primary_shape(grid: ColoredGrid) -> Set[Tuple[int, int]]:
        largest_region = max(
            (region for color in range(1, 10) for region in grid.find_connected_regions(color)),
            key=len, default=[]
        )
        return set(largest_region)

    def get_transformations(shape: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        transformations = []
        
        # Original shape
        transformations.append(shape)
        
        # Rotations
        for _ in range(3):
            shape = {(-c, r) for r, c in shape}
            transformations.append(shape)
        
        # Mirrors
        mirror_h = {(r, -c) for r, c in shape}
        mirror_v = {(-r, c) for r, c in shape}
        transformations.extend([mirror_h, mirror_v])
        
        # Rotations of mirrors
        for _ in range(3):
            mirror_h = {(-c, r) for r, c in mirror_h}
            mirror_v = {(-c, r) for r, c in mirror_v}
            transformations.extend([mirror_h, mirror_v])
        
        return transformations

    def apply_shape(grid: List[List[int]], shape: Set[Tuple[int, int]], offset_r: int, offset_c: int, color: int) -> int:
        rows, cols = len(grid), len(grid[0])
        score = 0
        for r, c in shape:
            grid_r, grid_c = offset_r + r, offset_c + c
            if 0 <= grid_r < rows and 0 <= grid_c < cols:
                if grid[grid_r][grid_c] == 0:
                    grid[grid_r][grid_c] = color
                    score += 1
        return score

    primary_shape = find_primary_shape(input_grid)
    output_grid = copy.deepcopy(input_grid.values)
    rows, cols = input_grid.get_dimensions()

    isolated_squares = [
        (r, c, input_grid.values[r][c])
        for r in range(rows)
        for c in range(cols)
        if input_grid.values[r][c] != 0 and (r, c) not in primary_shape
    ]

    transformations = get_transformations(primary_shape)

    for isolated_r, isolated_c, isolated_color in isolated_squares:
        best_score = 0
        best_placement = None

        for transform in transformations:
            min_r = min(r for r, _ in transform)
            min_c = min(c for _, c in transform)
            max_r = max(r for r, _ in transform)
            max_c = max(c for _, c in transform)

            for offset_r in range(-max_r, rows - min_r):
                for offset_c in range(-max_c, cols - min_c):
                    if (isolated_r - offset_r, isolated_c - offset_c) in transform:
                        temp_grid = copy.deepcopy(output_grid)
                        score = apply_shape(temp_grid, transform, offset_r, offset_c, isolated_color)
                        if score > best_score:
                            best_score = score
                            best_placement = (transform, offset_r, offset_c)

        if best_placement:
            transform, offset_r, offset_c = best_placement
            apply_shape(output_grid, transform, offset_r, offset_c, isolated_color)

    return ColoredGrid(values=output_grid)
