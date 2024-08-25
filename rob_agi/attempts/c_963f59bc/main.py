from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_963f59bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the primary shape and replicating it
    for each isolated square of a different color. The replicated shapes are
    rotated, mirrored, and positioned based on the location of the isolated squares.
    
    1. Identifies the primary shape (largest non-black connected region)
    2. Finds isolated squares of different colors
    3. For each isolated square, applies a transformed version of the primary shape:
       - Rotations (0°, 90°, 180°, 270°) and mirrors (horizontal, vertical)
       - Aligns the core of the transformed shape with the isolated cell
       - Allows overflow beyond grid boundaries if necessary
    4. Returns the modified grid with all transformations applied
    """
    def find_primary_shape(grid: ColoredGrid) -> List[List[int]]:
        largest_region = max(
            (region for color in range(1, 10) for region in grid.find_connected_regions(color)),
            key=len, default=[]
        )
        if not largest_region:
            return [[]]
        color = grid.values[largest_region[0][0]][largest_region[0][1]]
        min_r = min(r for r, c in largest_region)
        max_r = max(r for r, c in largest_region)
        min_c = min(c for r, c in largest_region)
        max_c = max(c for r, c in largest_region)
        return [[color if (r, c) in largest_region else 0 
                 for c in range(min_c, max_c + 1)]
                for r in range(min_r, max_r + 1)]

    def rotate_90(shape: List[List[int]]) -> List[List[int]]:
        return [list(row) for row in zip(*shape[::-1])]

    def mirror_horizontal(shape: List[List[int]]) -> List[List[int]]:
        return [row[::-1] for row in shape]

    def mirror_vertical(shape: List[List[int]]) -> List[List[int]]:
        return shape[::-1]

    def find_core(shape: List[List[int]]) -> Tuple[int, int]:
        rows, cols = len(shape), len(shape[0])
        non_zero = [(r, c) for r in range(rows) for c in range(cols) if shape[r][c] != 0]
        return sum(r for r, _ in non_zero) // len(non_zero), sum(c for _, c in non_zero) // len(non_zero)

    def apply_shape(grid: List[List[int]], shape: List[List[int]], center_r: int, center_c: int):
        shape_core_r, shape_core_c = find_core(shape)
        height, width = len(shape), len(shape[0])
        for i in range(height):
            for j in range(width):
                grid_r = center_r + (i - shape_core_r)
                grid_c = center_c + (j - shape_core_c)
                if 0 <= grid_r < len(grid) and 0 <= grid_c < len(grid[0]):
                    if shape[i][j] != 0:
                        grid[grid_r][grid_c] = shape[i][j]

    primary_shape = find_primary_shape(input_grid)
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and input_grid.values[r][c] != primary_shape[0][0]:
                isolated_color = input_grid.values[r][c]
                transformations = [
                    lambda s: s,  # no transformation
                    rotate_90,
                    lambda s: rotate_90(rotate_90(s)),  # 180 degrees
                    lambda s: rotate_90(rotate_90(rotate_90(s))),  # 270 degrees
                    mirror_horizontal,
                    mirror_vertical,
                    lambda s: rotate_90(mirror_horizontal(s)),
                    lambda s: rotate_90(mirror_vertical(s))
                ]
                
                best_score = -1
                best_transformed_shape = None
                
                for transform in transformations:
                    transformed_shape = [
                        [isolated_color if cell != 0 else 0 for cell in row]
                        for row in transform(primary_shape)
                    ]
                    temp_grid = [row[:] for row in output_grid.values]
                    apply_shape(temp_grid, transformed_shape, r, c)
                    score = sum(sum(1 for cell in row if cell == isolated_color) for row in temp_grid)
                    if score > best_score:
                        best_score = score
                        best_transformed_shape = transformed_shape

                if best_transformed_shape:
                    apply_shape(output_grid.values, best_transformed_shape, r, c)

    return output_grid
