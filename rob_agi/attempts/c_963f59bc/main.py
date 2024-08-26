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
       - Aligns key points of the transformed shape with the isolated square
       - Maximizes the number of filled squares within grid boundaries
    4. Applies transformations in order of best fit
    5. Preserves original elements and grid boundaries
    6. Returns the modified grid with all valid transformations applied
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

    def find_key_points(shape: List[List[int]]) -> List[Tuple[int, int]]:
        rows, cols = len(shape), len(shape[0])
        non_zero = [(r, c) for r in range(rows) for c in range(cols) if shape[r][c] != 0]
        return non_zero

    def apply_shape(grid: List[List[int]], shape: List[List[int]], offset_r: int, offset_c: int, color: int):
        height, width = len(shape), len(shape[0])
        rows, cols = len(grid), len(grid[0])
        for i in range(height):
            for j in range(width):
                grid_r, grid_c = offset_r + i, offset_c + j
                if 0 <= grid_r < rows and 0 <= grid_c < cols:
                    if shape[i][j] != 0 and grid[grid_r][grid_c] == 0:
                        grid[grid_r][grid_c] = color

    primary_shape = find_primary_shape(input_grid)
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    isolated_squares = []
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and input_grid.values[r][c] != primary_shape[0][0]:
                isolated_squares.append((r, c, input_grid.values[r][c]))

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

    for isolated_r, isolated_c, isolated_color in isolated_squares:
        best_score = -1
        best_transformation = None
        best_offset = None

        for transform in transformations:
            transformed_shape = transform(primary_shape)
            key_points = find_key_points(transformed_shape)

            for key_r, key_c in key_points:
                offset_r = isolated_r - key_r
                offset_c = isolated_c - key_c

                temp_grid = [row[:] for row in output_grid.values]
                apply_shape(temp_grid, transformed_shape, offset_r, offset_c, isolated_color)

                score = sum(sum(1 for cell in row if cell == isolated_color) for row in temp_grid)
                if score > best_score:
                    best_score = score
                    best_transformation = transformed_shape
                    best_offset = (offset_r, offset_c)

        if best_transformation:
            apply_shape(output_grid.values, best_transformation, best_offset[0], best_offset[1], isolated_color)

    return output_grid
