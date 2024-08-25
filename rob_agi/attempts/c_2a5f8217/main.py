from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, FrozenSet

def solve_2a5f8217(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying unique shapes,
    finding the highest color for each shape, and applying these colors
    consistently across the grid.

    1. Identify unique shapes in the input grid.
    2. Find the highest color value for each unique shape.
    3. Create a new grid and apply the highest color to all instances of each shape.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with colors updated according to the rule.
    """
    def normalize_shape(coords: List[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
        min_x = min(x for x, y in coords)
        min_y = min(y for x, y in coords)
        return frozenset((x - min_x, y - min_y) for x, y in coords)

    shapes: Dict[FrozenSet[Tuple[int, int]], List[Tuple[int, List[Tuple[int, int]]]]] = {}
    for color in range(1, 10):  # Assuming colors are 1-9
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            shape = normalize_shape(region)
            if shape not in shapes:
                shapes[shape] = []
            shapes[shape].append((color, region))

    shape_highest_color = {
        shape: max(color for color, _ in instances)
        for shape, instances in shapes.items()
    }

    new_grid = ColoredGrid(values=[
        [0 for _ in range(input_grid.num_cols)]
        for _ in range(input_grid.num_rows)
    ])

    for shape, instances in shapes.items():
        new_color = shape_highest_color[shape]
        for _, region in instances:
            for x, y in region:
                new_grid.values[y][x] = new_color

    return new_grid
