from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import math

def solve_55783887(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored dots with diagonal paths.
    
    The solution follows these steps:
    1. Identify the background color and all non-background colored dots.
    2. For each color, find the longest diagonal between any two dots.
    3. Draw the longest diagonal for each color.
    4. Connect any remaining dots to the nearest point on an existing line of the same color.
    5. Optimize the solution by extending lines where possible.
    
    The result maximizes diagonal path lengths while ensuring all dots of the same color are connected.
    """
    output_grid = input_grid.deep_copy()
    background_color = find_background_color(input_grid)
    colored_dots = find_colored_dots(input_grid, background_color)
    
    for color, dots in colored_dots.items():
        if len(dots) > 1:
            longest_diagonal = find_longest_diagonal(dots)
            draw_diagonal(output_grid, longest_diagonal, color)
            connect_remaining_dots(output_grid, dots, longest_diagonal, color)
    
    return output_grid

def find_background_color(grid: ColoredGrid) -> int:
    return max(grid.get_color_frequencies(), key=grid.get_color_frequencies().get)

def find_colored_dots(grid: ColoredGrid, background_color: int) -> Dict[int, List[Tuple[int, int]]]:
    dots = {}
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            color = grid.get_cell(r, c)
            if color != background_color:
                if color not in dots:
                    dots[color] = []
                dots[color].append((r, c))
    return dots

def find_longest_diagonal(dots: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    longest_diagonal = []
    max_distance = 0
    for i, start in enumerate(dots):
        for end in dots[i+1:]:
            diagonal = get_diagonal_path(start, end)
            distance = len(diagonal)
            if distance > max_distance:
                max_distance = distance
                longest_diagonal = diagonal
    return longest_diagonal

def get_diagonal_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    r, c = start
    dr = 1 if end[0] > r else -1 if end[0] < r else 0
    dc = 1 if end[1] > c else -1 if end[1] < c else 0
    while (r, c) != end:
        path.append((r, c))
        r, c = r + dr, c + dc
    path.append(end)
    return path

def draw_diagonal(grid: ColoredGrid, diagonal: List[Tuple[int, int]], color: int):
    for r, c in diagonal:
        grid.set_cell(r, c, color)

def connect_remaining_dots(grid: ColoredGrid, dots: List[Tuple[int, int]], main_diagonal: List[Tuple[int, int]], color: int):
    for dot in dots:
        if dot not in main_diagonal:
            nearest_point = min(main_diagonal, key=lambda x: distance(dot, x))
            path = get_diagonal_path(dot, nearest_point)
            draw_diagonal(grid, path, color)

def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
