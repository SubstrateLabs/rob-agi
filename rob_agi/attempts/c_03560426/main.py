from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

from typing import List, Tuple, Dict
from collections import deque

class Shape:
    def __init__(self, color: int, coords: List[Tuple[int, int]], order: int):
        self.color = color
        self.coords = coords
        self.order = order

    def rotate(self):
        self.coords = [(y, -x) for x, y in self.coords]
        min_x = min(x for x, _ in self.coords)
        min_y = min(y for _, y in self.coords)
        self.coords = [(x - min_x, y - min_y) for x, y in self.coords]

class Column:
    def __init__(self):
        self.shapes = []
        self.width = 0
        self.height = 0

    def add_shape(self, shape: Shape, rotated: bool):
        if rotated:
            shape.rotate()
        self.shapes.append(shape)
        self.width = max(self.width, max(x for x, _ in shape.coords) + 1)
        self.height += max(y for _, y in shape.coords) + 1

def solve_03560426(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by rearranging colored shapes into a compact arrangement.
    
    1. Extracts shapes from bottom to top, left to right.
    2. Creates columns of shapes, rotating only when necessary to fit.
    3. Aligns shapes within columns to the bottom.
    4. Moves all shapes in each column up as much as possible.
    5. Compacts columns horizontally to the left.
    6. Places shapes in the output grid according to their new positions.
    7. Fills remaining space with black (0).
    
    Returns the transformed grid with shapes arranged compactly in the top-left quadrant.
    """
    shapes = extract_shapes(input_grid)
    columns = create_columns(shapes)
    align_columns_to_top(columns)
    compact_columns_horizontally(columns)
    output_grid = create_output_grid(columns)
    return output_grid

def extract_shapes(grid: ColoredGrid) -> List[Shape]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    order = 0
    
    for r in range(rows-1, -1, -1):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                coords = bfs(grid, r, c, visited)
                shapes.append(Shape(grid.values[r][c], coords, order))
                order += 1
    
    return shapes

def bfs(grid: ColoredGrid, start_r: int, start_c: int, visited: set) -> List[Tuple[int, int]]:
    queue = deque([(start_r, start_c)])
    shape = []
    color = grid.values[start_r][start_c]
    rows, cols = grid.get_dimensions()
    min_r, min_c = start_r, start_c
    
    while queue:
        r, c = queue.popleft()
        if (r, c) not in visited and grid.values[r][c] == color:
            visited.add((r, c))
            min_r = min(min_r, r)
            min_c = min(min_c, c)
            shape.append((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    queue.append((nr, nc))
    
    return [(r - min_r, c - min_c) for r, c in shape]  # Store relative coordinates

def create_columns(shapes: List[Shape]) -> List[Column]:
    columns = []
    for shape in shapes:
        placed = False
        for column in columns:
            if column.width + max(x for x, _ in shape.coords) + 1 <= 10:
                column.add_shape(shape, False)
                placed = True
                break
            elif column.width + max(y for _, y in shape.coords) + 1 <= 10:
                column.add_shape(shape, True)
                placed = True
                break
        if not placed:
            new_column = Column()
            new_column.add_shape(shape, False)
            columns.append(new_column)
    return columns

def align_columns_to_top(columns: List[Column]) -> None:
    for column in columns:
        max_height = sum(max(y for _, y in shape.coords) + 1 for shape in column.shapes)
        current_height = 0
        for shape in column.shapes:
            shape_height = max(y for _, y in shape.coords) + 1
            shape.coords = [(x, y + (10 - max_height) + current_height) for x, y in shape.coords]
            current_height += shape_height

def compact_columns_horizontally(columns: List[Column]) -> None:
    current_x = 0
    for column in columns:
        for shape in column.shapes:
            shape.coords = [(x + current_x, y) for x, y in shape.coords]
        current_x += column.width

def create_output_grid(columns: List[Column]) -> ColoredGrid:
    output_grid = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    for column in columns:
        for shape in column.shapes:
            for x, y in shape.coords:
                output_grid.values[y][x] = shape.color
    return output_grid
