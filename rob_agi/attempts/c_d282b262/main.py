from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

class Shape:
    def __init__(self, colors: List[List[int]], top: int, left: int):
        self.colors = colors
        self.height = len(colors)
        self.width = len(colors[0])
        self.top = top
        self.left = left

    def place(self, grid: List[List[int]], top: int, left: int):
        for i in range(self.height):
            for j in range(self.width):
                grid[top + i][left + j] = self.colors[i][j]

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero shapes to the right side of the grid.
    
    1. Identifies and extracts all non-zero shapes from the input grid.
    2. Sorts shapes based on their original vertical position (top to bottom), then left to right.
    3. Places shapes on the right side of a new grid, starting from the top-right corner,
       moving left and down as needed to fit all shapes without overlap.
    4. Maintains a 3-column gap on the right side of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with shapes moved to the right side.
    """
    def find_shapes(grid: List[List[int]]) -> List[Shape]:
        shapes = []
        visited = set()
        rows, cols = len(grid), len(grid[0])
        
        def dfs(r: int, c: int, color: int) -> List[List[int]]:
            shape = []
            stack = [(r, c)]
            min_r, min_c = r, c
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and grid[curr_r][curr_c] == color:
                    visited.add((curr_r, curr_c))
                    shape.append((curr_r, curr_c))
                    min_r, min_c = min(min_r, curr_r), min(min_c, curr_c)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return shape, min_r, min_c

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid[r][c] != 0:
                    shape_coords, top, left = dfs(r, c, grid[r][c])
                    if shape_coords:
                        shape_grid = [[0 for _ in range(cols)] for _ in range(rows)]
                        for sr, sc in shape_coords:
                            shape_grid[sr - top][sc - left] = grid[sr][sc]
                        shapes.append(Shape(shape_grid, top, left))

        return sorted(shapes, key=lambda s: (s.top, s.left))

    shapes = find_shapes(input_grid.values)
    output_grid = [[0 for _ in range(15)] for _ in range(15)]
    
    current_col = 11
    for shape in shapes:
        placed = False
        while not placed and current_col >= 0:
            for row in range(15 - shape.height + 1):
                if all(output_grid[r][c] == 0 
                       for r in range(row, row + shape.height) 
                       for c in range(current_col, min(15, current_col + shape.width))):
                    shape.place(output_grid, row, current_col)
                    placed = True
                    break
            if not placed:
                current_col -= 1

    return ColoredGrid(values=output_grid)
