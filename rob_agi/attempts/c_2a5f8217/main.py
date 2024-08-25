from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
from collections import deque

def solve_2a5f8217(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by identifying shapes and propagating colors.

    1. Identify shapes in the input grid.
    2. Build a connection graph between shapes based on adjacency.
    3. Propagate colors from higher-valued shapes to lower-valued connected shapes.
    4. Apply the color transformations to create a new grid.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with colors updated according to the rule.
    """
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

    def are_adjacent(shape1: Set[Tuple[int, int]], shape2: Set[Tuple[int, int]]) -> bool:
        for x1, y1 in shape1:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                if (x1 + dx, y1 + dy) in shape2:
                    return True
        return False

    rows, cols = input_grid.get_dimensions()
    shapes: List[Tuple[int, Set[Tuple[int, int]]]] = []
    visited = set()

    # Identify shapes
    for x in range(rows):
        for y in range(cols):
            if (x, y) not in visited and input_grid.values[x][y] != 0:
                shape = find_shape(x, y, input_grid.values[x][y])
                shapes.append((input_grid.values[x][y], shape))
                visited.update(shape)

    # Build connection graph
    graph: Dict[int, List[int]] = {i: [] for i in range(len(shapes))}
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if are_adjacent(shapes[i][1], shapes[j][1]):
                graph[i].append(j)
                graph[j].append(i)

    # Propagate colors
    shapes.sort(reverse=True)  # Sort by color value, highest first
    new_colors: Dict[int, int] = {}
    for i, (color, shape) in enumerate(shapes):
        if i not in new_colors:
            new_colors[i] = color
        queue = deque([i])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in new_colors and shapes[neighbor][0] < new_colors[node]:
                    new_colors[neighbor] = new_colors[node]
                    queue.append(neighbor)

    # Apply transformations
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for i, (_, shape) in enumerate(shapes):
        for x, y in shape:
            new_grid[x][y] = new_colors[i]

    return ColoredGrid(values=new_grid)
