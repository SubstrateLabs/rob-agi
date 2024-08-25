from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
from collections import deque

def solve_2a5f8217(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying shapes and updating their colors
    based on the highest adjacent color value, without expanding the shapes.

    1. Identify shapes in the input grid.
    2. Create an adjacency map for shapes.
    3. Determine color changes based on adjacent higher-valued shapes.
    4. Apply color changes to the shapes.
    5. Create and return the transformed grid.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with colors updated.
    """
    rows, cols = input_grid.get_dimensions()
    
    def find_shape(start_x: int, start_y: int, color: int) -> Set[Tuple[int, int]]:
        shape = set()
        queue = deque([(start_x, start_y)])
        while queue:
            x, y = queue.popleft()
            if (x, y) not in shape and 0 <= x < rows and 0 <= y < cols and input_grid.values[x][y] == color:
                shape.add((x, y))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((x + dx, y + dy))
        return shape

    def get_adjacent_cells(shape: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        adjacent = set()
        for x, y in shape:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in shape:
                    adjacent.add((nx, ny))
        return adjacent

    # Identify shapes
    shapes = []
    visited = set()
    for x in range(rows):
        for y in range(cols):
            if (x, y) not in visited and input_grid.values[x][y] != 0:
                shape = find_shape(x, y, input_grid.values[x][y])
                shapes.append((input_grid.values[x][y], shape))
                visited.update(shape)

    # Create adjacency map
    adjacency_map = {}
    for i, (color, shape) in enumerate(shapes):
        adjacent_cells = get_adjacent_cells(shape)
        adjacency_map[i] = [(j, other_color) for j, (other_color, other_shape) in enumerate(shapes)
                            if i != j and any(cell in other_shape for cell in adjacent_cells)]

    # Determine color changes
    new_colors = {}
    for i, (color, _) in enumerate(shapes):
        adjacent_colors = [other_color for _, other_color in adjacency_map[i]]
        new_colors[i] = max([color] + adjacent_colors)

    # Apply color changes
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for i, (_, shape) in enumerate(shapes):
        new_color = new_colors[i]
        for x, y in shape:
            new_grid[x][y] = new_color

    return ColoredGrid(values=new_grid)
