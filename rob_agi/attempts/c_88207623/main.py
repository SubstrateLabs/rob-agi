from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque

def solve_88207623(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by expanding unique color pixels around yellow shapes.
    
    1. Identifies yellow shapes with their red borders.
    2. Locates single pixels of unique colors.
    3. Associates each unique color pixel with its nearest yellow shape.
    4. Expands unique color pixels to form territories around their associated yellow shapes.
    5. Preserves original yellow shapes, red borders, and unique color pixel positions.
    
    The expansion fills available black space in all directions, stopping at existing colors,
    other expanded territories, or grid edges. Closer unique color pixels to a yellow shape
    take precedence in expansion over more distant ones.
    """
    def find_yellow_shapes_and_borders(grid: ColoredGrid) -> List[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
        shapes = []
        rows, cols = grid.get_dimensions()
        visited = set()
        
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] == 4:
                    shape = set()
                    border = set()
                    queue = deque([(r, c)])
                    
                    while queue:
                        cr, cc = queue.popleft()
                        if (cr, cc) in visited:
                            continue
                        visited.add((cr, cc))
                        
                        if grid.values[cr][cc] == 4:
                            shape.add((cr, cc))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if grid.values[nr][nc] == 4:
                                        queue.append((nr, nc))
                                    elif grid.values[nr][nc] == 2:
                                        border.add((nr, nc))
                    
                    shapes.append((shape, border))
        
        return shapes

    def find_unique_color_pixels(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        unique_pixels = []
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                color = grid.values[r][c]
                if color not in {0, 2, 4}:
                    unique_pixels.append((r, c, color))
        return unique_pixels

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def associate_pixels_to_shapes(pixels: List[Tuple[int, int, int]], shapes: List[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]) -> Dict[int, List[Tuple[int, int, int]]]:
        pixel_shape_mapping = defaultdict(list)
        for pixel in pixels:
            distances = [min(manhattan_distance(pixel[:2], cell) for cell in shape[0]) for shape in shapes]
            nearest_shape = distances.index(min(distances))
            pixel_shape_mapping[nearest_shape].append((pixel, min(distances)))
        return {k: sorted(v, key=lambda x: x[1]) for k, v in pixel_shape_mapping.items()}

    def expand_territory(grid: ColoredGrid, start: Tuple[int, int], color: int, expanded: Set[Tuple[int, int]], shape_set: Set[Tuple[int, int]]) -> None:
        rows, cols = grid.get_dimensions()
        queue = deque([start])
        visited = set()

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or (r, c) in expanded or grid.values[r][c] != 0 or (r, c) in shape_set:
                continue

            grid.values[r][c] = color
            visited.add((r, c))
            expanded.add((r, c))

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))

    # Main solving process
    result_grid = input_grid.deep_copy()
    yellow_shapes_and_borders = find_yellow_shapes_and_borders(result_grid)
    unique_color_pixels = find_unique_color_pixels(input_grid)
    pixel_shape_mapping = associate_pixels_to_shapes(unique_color_pixels, yellow_shapes_and_borders)

    shape_set = set()
    for shape, border in yellow_shapes_and_borders:
        shape_set.update(shape)
        shape_set.update(border)

    expanded = set()
    for shape_index, pixels in pixel_shape_mapping.items():
        for (r, c, color), _ in pixels:
            if (r, c) not in expanded:
                expand_territory(result_grid, (r, c), color, expanded, shape_set)

    # Preserve original shapes, borders, and unique color pixels
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.values[r][c] != 0:
                result_grid.values[r][c] = input_grid.values[r][c]

    return result_grid
