from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

class Shape:
    def __init__(self, colors: List[List[int]], top: int, left: int):
        self.colors = colors
        self.height = len(colors)
        self.width = len(colors[0])
        self.top = top
        self.left = left

    def can_place_at(self, grid: List[List[int]], top: int, left: int) -> bool:
        if top + self.height > len(grid) or left + self.width > len(grid[0]):
            return False
        return all(grid[top + i][left + j] == 0
                   for i in range(self.height)
                   for j in range(self.width))

    def place(self, grid: List[List[int]], top: int, left: int):
        for i in range(self.height):
            for j in range(self.width):
                if left + j < len(grid[0]):  # Ensure we don't go out of bounds
                    grid[top + i][left + j] = self.colors[i][j]

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero shapes to the right side of the grid.
    
    1. Identifies and extracts all non-zero shapes from the input grid.
    2. Sorts shapes based on their original vertical position (top to bottom), then left to right.
    3. Places shapes on the right side of a new grid, starting from the top-right corner,
       moving left and down as needed to fit all shapes without overlap.
    4. Maintains a 3-column gap on the right side of the grid.
    5. Aligns shapes vertically and compacts them horizontally when possible.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with shapes moved to the right side.
    """
    def find_shapes(grid: List[List[int]]) -> List[Shape]:
        shapes = []
        visited = set()
        rows, cols = len(grid), len(grid[0])
        
        def dfs(r: int, c: int, color: int) -> Tuple[List[Tuple[int, int]], int, int]:
            shape = []
            stack = [(r, c)]
            min_r, min_c = r, c
            max_r, max_c = r, c
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and grid[curr_r][curr_c] == color:
                    visited.add((curr_r, curr_c))
                    shape.append((curr_r, curr_c))
                    min_r, min_c = min(min_r, curr_r), min(min_c, curr_c)
                    max_r, max_c = max(max_r, curr_r), max(max_c, curr_c)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return shape, min_r, min_c, max_r - min_r + 1, max_c - min_c + 1

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] != 0:
                    shape_coords, top, left, height, width = dfs(r, c, grid[r][c])
                    if shape_coords:
                        shape_grid = [[0 for _ in range(width)] for _ in range(height)]
                        for sr, sc in shape_coords:
                            shape_grid[sr - top][sc - left] = grid[sr][sc]
                        shapes.append(Shape(shape_grid, top, left))

        return sorted(shapes, key=lambda s: (s.top, s.left))

    def align_vertically(grid: List[List[int]], shapes: List[Shape]):
        for i in range(len(shapes) - 1):
            current_shape = shapes[i]
            next_shape = shapes[i + 1]
            if next_shape.left > current_shape.left + current_shape.width:
                # Move the next shape up if there's space
                while next_shape.top > 0 and all(grid[next_shape.top - 1][c] == 0 for c in range(next_shape.left, next_shape.left + next_shape.width)):
                    for r in range(next_shape.height):
                        for c in range(next_shape.width):
                            grid[next_shape.top + r - 1][next_shape.left + c] = grid[next_shape.top + r][next_shape.left + c]
                            grid[next_shape.top + r][next_shape.left + c] = 0
                    next_shape.top -= 1

    def compact_horizontally(grid: List[List[int]], shapes: List[Shape]):
        for col in range(len(grid[0]) - 4, -1, -1):  # Start from the 4th column from the right
            can_move = True
            for shape in shapes:
                if shape.left <= col:
                    if not shape.can_place_at(grid, shape.top, shape.left + 1):
                        can_move = False
                        break
            if can_move:
                for shape in shapes:
                    if shape.left <= col:
                        shape.place(grid, shape.top, shape.left + 1)
                        for r in range(shape.height):
                            grid[shape.top + r][shape.left] = 0
                        shape.left += 1
            else:
                break

    shapes = find_shapes(input_grid.values)
    output_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    current_col = 11
    for shape in shapes:
        placed = False
        while not placed and current_col >= 0:
            for row in range(15 - shape.height + 1):
                if shape.can_place_at(output_grid, row, current_col):
                    shape.place(output_grid, row, current_col)
                    placed = True
                    break
            if not placed:
                current_col -= 1

    align_vertically(output_grid, shapes)
    compact_horizontally(output_grid, shapes)

    return ColoredGrid(values=output_grid)
