from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_55783887(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by connecting colored dots with diagonal paths.
    
    The solution follows these steps:
    1. Identify the background color and all non-background colored dots.
    2. For each color:
       a. Sort the dot positions from top-left to bottom-right.
       b. Create a path through all dots, extending diagonally beyond the first and last dots.
       c. Connect the points in the path using diagonal movements, creating zigzags when necessary.
    3. Draw all paths on the output grid.
    4. Ensure all original dots are preserved.
    
    The result creates continuous diagonal lines for each color, extending diagonally beyond the dots,
    while connecting all dots of the same color and allowing intersections between different colors.
    Single dots are preserved without creating paths.
    """
    background_color = find_background_color(input_grid)
    colored_dots = find_colored_dots(input_grid, background_color)
    output_grid = ColoredGrid(values=[[background_color for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    
    for color, dots in colored_dots.items():
        if len(dots) > 1:
            path = create_extended_path(dots, input_grid.num_rows, input_grid.num_cols)
            connected_path = connect_points(path)
            draw_path(output_grid, connected_path, color)
        else:
            # For single dots, just preserve them without creating a path
            r, c = dots[0]
            output_grid.set_cell(r, c, color)
    
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

def create_extended_path(dots: List[Tuple[int, int]], max_row: int, max_col: int) -> List[Tuple[int, int]]:
    sorted_dots = sorted(dots)
    first_dot, last_dot = sorted_dots[0], sorted_dots[-1]
    
    # Extend diagonally before the first dot
    start = extend_path(first_dot, (-1, -1), max_row, max_col)
    
    # Extend diagonally after the last dot
    end = extend_path(last_dot, (1, 1), max_row, max_col)
    
    return [start] + sorted_dots + [end]

def extend_path(dot: Tuple[int, int], direction: Tuple[int, int], max_row: int, max_col: int) -> Tuple[int, int]:
    r, c = dot
    dr, dc = direction
    while 0 <= r + dr < max_row and 0 <= c + dc < max_col:
        r += dr
        c += dc
    return (r, c)

def connect_points(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    connected_path = []
    for i in range(len(path) - 1):
        connected_path.extend(get_diagonal_path(path[i], path[i+1]))
    return connected_path

def get_diagonal_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    current = start
    while current != end:
        path.append(current)
        dr = sign(end[0] - current[0])
        dc = sign(end[1] - current[1])
        if dr != 0 and dc != 0:
            current = (current[0] + dr, current[1] + dc)
        elif dr != 0:
            current = (current[0] + dr, current[1])
        else:
            current = (current[0], current[1] + dc)
    path.append(end)
    return path

def sign(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0

def draw_path(grid: ColoredGrid, path: List[Tuple[int, int]], color: int):
    for r, c in path:
        grid.set_cell(r, c, color)
