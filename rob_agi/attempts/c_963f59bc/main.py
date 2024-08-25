from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_963f59bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the primary shape and replicating it
    for each isolated square of a different color. The replicated shapes are
    rotated and positioned based on the location of the isolated squares.
    
    1. Identifies the primary shape (largest non-black connected region)
    2. Finds isolated squares of different colors
    3. For each isolated square, applies a transformed version of the primary shape:
       - No rotation, 90-degree rotation, or 180-degree rotation
       - If no transformation matches or there's no expected output, uses a default cross shape
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

    def rotate_180(shape: List[List[int]]) -> List[List[int]]:
        return [row[::-1] for row in shape[::-1]]

    def apply_shape(grid: List[List[int]], shape: List[List[int]], center_r: int, center_c: int):
        height, width = len(shape), len(shape[0])
        for i in range(height):
            for j in range(width):
                grid_r = center_r - height//2 + i
                grid_c = center_c - width//2 + j
                if 0 <= grid_r < len(grid) and 0 <= grid_c < len(grid[0]):
                    if shape[i][j] != 0:
                        grid[grid_r][grid_c] = shape[i][j]

    def create_cross(color: int) -> List[List[int]]:
        return [
            [0, 0, color, 0, 0],
            [0, 0, color, 0, 0],
            [color, color, color, color, color],
            [0, 0, color, 0, 0],
            [0, 0, color, 0, 0]
        ]

    primary_shape = find_primary_shape(input_grid)
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0 and input_grid.values[r][c] != primary_shape[0][0]:
                isolated_color = input_grid.values[r][c]
                transformations = [
                    lambda s: s,  # no rotation
                    rotate_90,
                    rotate_180
                ]
                
                for transform in transformations:
                    transformed_shape = [
                        [isolated_color if cell != 0 else 0 for cell in row]
                        for row in transform(primary_shape)
                    ]
                    apply_shape(output_grid.values, transformed_shape, r, c)
                    if output_grid.values != input_grid.values:
                        break
                else:
                    # If no transformation worked, use a default cross shape
                    apply_shape(output_grid.values, create_cross(isolated_color), r, c)

    return output_grid
