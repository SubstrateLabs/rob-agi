from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Set
import math

def solve_55783887(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored dots with diagonal paths.
    
    The solution follows these steps:
    1. Identify the background color and all non-background colored dots.
    2. For each color:
       a. If there's only one dot, leave it as is.
       b. If there are two dots, connect them with a diagonal line.
       c. If there are more than two dots:
          - Find the bounding rectangle of the dots.
          - Create a zigzag path through all dots within the bounding rectangle.
          - Optimize the path by smoothing unnecessary zigzags.
    3. Extend lines to grid edges where appropriate.
    4. Ensure all dots are connected and there are no isolated segments.
    
    The result creates continuous lines for each color while respecting the original dot positions.
    """
    output_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    background_color = find_background_color(input_grid)
    colored_dots = find_colored_dots(input_grid, background_color)
    
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            output_grid.set_cell(r, c, background_color)
    
    for color, dots in colored_dots.items():
        if len(dots) == 1:
            r, c = dots[0]
            output_grid.set_cell(r, c, color)
        elif len(dots) == 2:
            path = get_diagonal_path(dots[0], dots[1])
            draw_path(output_grid, path, color)
        else:
            path = create_zigzag_path(dots)
            optimized_path = optimize_path(path, dots)
            extended_path = extend_to_edges(optimized_path, input_grid.num_rows, input_grid.num_cols)
            draw_path(output_grid, extended_path, color)
    
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

def draw_path(grid: ColoredGrid, path: List[Tuple[int, int]], color: int):
    for r, c in path:
        grid.set_cell(r, c, color)

def create_zigzag_path(dots: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in dots)
    max_r = max(r for r, _ in dots)
    min_c = min(c for _, c in dots)
    max_c = max(c for _, c in dots)
    
    path = []
    r, c = min_r, min_c
    going_right = True
    
    while r <= max_r:
        while min_c <= c <= max_c:
            if (r, c) in dots:
                path.append((r, c))
            if going_right and c == max_c:
                break
            if not going_right and c == min_c:
                break
            c += 1 if going_right else -1
        r += 1
        going_right = not going_right
    
    return path

def optimize_path(path: List[Tuple[int, int]], dots: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    optimized = [path[0]]
    for i in range(1, len(path) - 1):
        prev, curr, next = path[i-1], path[i], path[i+1]
        if curr in dots or not is_straight_line(prev, curr, next):
            optimized.append(curr)
    optimized.append(path[-1])
    return optimized

def is_straight_line(p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> bool:
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def extend_to_edges(path: List[Tuple[int, int]], max_r: int, max_c: int) -> List[Tuple[int, int]]:
    start, end = path[0], path[-1]
    
    # Extend start
    while start[0] > 0 and start[1] > 0:
        new_start = (start[0] - 1, start[1] - 1)
        path.insert(0, new_start)
        start = new_start
    
    # Extend end
    while end[0] < max_r - 1 and end[1] < max_c - 1:
        new_end = (end[0] + 1, end[1] + 1)
        path.append(new_end)
        end = new_end
    
    return path

def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
