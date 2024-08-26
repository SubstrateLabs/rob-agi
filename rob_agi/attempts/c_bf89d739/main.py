from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import statistics

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function identifies red dots, creates a vertical spine through the optimal column,
    connects all red dots to the spine or to each other, and optimizes the connections
    to form a minimal tree-like structure.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming an optimized tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = find_red_dots(input_grid)
    
    if not red_dots:
        return result_grid

    spine_col = find_optimal_spine(red_dots)
    create_vertical_spine(result_grid, red_dots, spine_col)
    connect_dots_optimally(result_grid, red_dots, spine_col)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def find_optimal_spine(red_dots: List[Tuple[int, int]]) -> int:
    x_coords = [c for _, c in red_dots]
    return round(statistics.mean(x_coords))

def create_vertical_spine(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    min_row = min(r for r, _ in red_dots)
    max_row = max(r for r, _ in red_dots)
    draw_line(grid, (min_row, spine_col), (max_row, spine_col), True)

def connect_dots_optimally(grid: ColoredGrid, red_dots: List[Tuple[int, int]], spine_col: int):
    sorted_dots = sorted(red_dots, key=lambda x: (x[0], abs(x[1] - spine_col)))  # Sort by row, then by distance to spine
    connected = set()

    for i, dot in enumerate(sorted_dots):
        if dot not in connected:
            connect_dot(grid, dot, sorted_dots[i+1:], spine_col, connected)

def connect_dot(grid: ColoredGrid, dot: Tuple[int, int], remaining_dots: List[Tuple[int, int]], spine_col: int, connected: set):
    connected.add(dot)
    r, c = dot

    # Try to connect to the closest dot
    closest_dot = min(remaining_dots, key=lambda x: manhattan_distance(dot, x), default=None)
    if closest_dot and manhattan_distance(dot, closest_dot) <= abs(c - spine_col):
        draw_line(grid, dot, closest_dot, False)
        connect_dot(grid, closest_dot, [d for d in remaining_dots if d != closest_dot], spine_col, connected)
    else:
        # Connect to spine if no close dot
        draw_line(grid, (r, c), (r, spine_col), False)

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int], is_vertical: bool):
    y1, x1 = start
    y2, x2 = end
    
    if is_vertical:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if grid.values[y][x1] == 0:  # Only fill black cells
                grid.values[y][x1] = 3  # Green
    else:
        if y1 == y2:  # Horizontal line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                if grid.values[y1][x] == 0:  # Only fill black cells
                    grid.values[y1][x] = 3  # Green
        else:  # Diagonal line
            x, y = x1, y1
            dx = 1 if x2 > x1 else -1
            dy = 1 if y2 > y1 else -1
            while (x, y) != (x2, y2):
                if grid.values[y][x] == 0:  # Only fill black cells
                    grid.values[y][x] = 3  # Green
                if x != x2:
                    x += dx
                if y != y2:
                    y += dy
            if grid.values[y2][x2] == 0:
                grid.values[y2][x2] = 3  # Ensure end point is marked
