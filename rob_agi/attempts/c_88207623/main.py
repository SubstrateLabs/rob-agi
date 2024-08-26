from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import defaultdict

def solve_88207623(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by expanding single pixels adjacent to main shapes.
    
    1. Identifies main shapes (yellow regions with red borders).
    2. Finds single pixels of unique colors near main shapes.
    3. Expands these pixels to form new borders around the main shapes.
    4. Preserves the original main shapes and their borders.
    
    The expansion fills available space in all directions, stopping at existing borders, 
    other colors, or grid edges. Expansions are based on proximity to the nearest main shape.
    Single pixels expand to form territories around their nearest yellow shape, with closer
    pixels taking precedence over more distant ones.
    """
    def find_main_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
        return grid.find_connected_regions(4)  # Yellow color

    def find_single_color_pixels(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        single_pixels = []
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                color = grid.values[r][c]
                if color not in {0, 2, 4}:  # Not black, red, or yellow
                    is_single = all(
                        grid.values[nr][nc] in {0, 2, 4}
                        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                        if 0 <= nr < rows and 0 <= nc < cols
                    )
                    if is_single:
                        single_pixels.append((r, c, color))
        return single_pixels

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def associate_pixels_to_shapes(pixels: List[Tuple[int, int, int]], shapes: List[List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int, int]]]:
        pixel_shape_mapping = defaultdict(list)
        for pixel in pixels:
            distances = [min(manhattan_distance(pixel[:2], cell) for cell in shape) for shape in shapes]
            nearest_shape = distances.index(min(distances))
            pixel_shape_mapping[nearest_shape].append((pixel, min(distances)))
        return {k: sorted(v, key=lambda x: x[1]) for k, v in pixel_shape_mapping.items()}

    def expand_territory(grid: ColoredGrid, start: Tuple[int, int], color: int, shape_set: Set[Tuple[int, int]]) -> None:
        rows, cols = grid.get_dimensions()
        queue = [start]
        visited = set()

        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited or grid.values[r][c] != 0 or (r, c) in shape_set:
                continue
        
            grid.values[r][c] = color
            visited.add((r, c))

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

    # Main solving process
    result_grid = input_grid.deep_copy()
    main_shapes = find_main_shapes(result_grid)
    single_color_pixels = find_single_color_pixels(input_grid)
    pixel_shape_mapping = associate_pixels_to_shapes(single_color_pixels, main_shapes)

    shape_set = set()
    for shape in main_shapes:
        shape_set.update(shape)
        for r, c in shape:
            if input_grid.values[r][c] == 2:  # Add red borders to shape_set
                shape_set.add((r, c))

    for shape_index, pixels in pixel_shape_mapping.items():
        for (r, c, color), _ in pixels:
            if result_grid.values[r][c] == 0:  # Only expand if the cell is still black
                expand_territory(result_grid, (r, c), color, shape_set)

    # Preserve original shapes and borders
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] in {2, 4}:  # Red border or yellow shape
                result_grid.values[r][c] = input_grid.values[r][c]

    return result_grid
