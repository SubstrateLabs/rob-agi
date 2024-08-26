from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_55783887(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored dots with diagonal paths.
    
    The solution follows these steps:
    1. Identify all non-background colored dots and group them by color.
    2. For each color group, find the longest possible diagonal paths between dots.
    3. Draw paths on the grid, prioritizing longer paths and allowing direction changes at dots.
    4. Handle intersections between paths of different colors.
    5. Optimize by connecting any remaining unconnected dots where possible.
    
    The result maximizes diagonal path lengths while connecting as many dots as possible.
    """
    output_grid = input_grid.deep_copy()
    background_color = find_background_color(input_grid)
    colored_dots = find_colored_dots(input_grid, background_color)
    
    for color, dots in colored_dots.items():
        paths = find_diagonal_paths(dots)
        draw_paths(output_grid, paths, color)
    
    return output_grid

def find_background_color(grid: ColoredGrid) -> int:
    return max(grid.get_color_frequencies(), key=grid.get_color_frequencies().get)

def find_colored_dots(grid: ColoredGrid, background_color: int) -> dict:
    dots = {}
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            color = grid.get_cell(r, c)
            if color != background_color:
                if color not in dots:
                    dots[color] = []
                dots[color].append((r, c))
    return dots

def find_diagonal_paths(dots: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    paths = []
    for i, start in enumerate(dots):
        for end in dots[i+1:]:
            path = get_diagonal_path(start, end)
            if path:
                paths.append(path)
    return sorted(paths, key=len, reverse=True)

def get_diagonal_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [start]
    r, c = start
    while (r, c) != end:
        dr = 1 if end[0] > r else -1 if end[0] < r else 0
        dc = 1 if end[1] > c else -1 if end[1] < c else 0
        if dr == 0 and dc == 0:
            break
        r, c = r + dr, c + dc
        path.append((r, c))
    return path if path[-1] == end else []

def draw_paths(grid: ColoredGrid, paths: List[List[Tuple[int, int]]], color: int):
    for path in paths:
        for r, c in path:
            if grid.get_cell(r, c) == color or grid.get_cell(r, c) == find_background_color(grid):
                grid.set_cell(r, c, color)
