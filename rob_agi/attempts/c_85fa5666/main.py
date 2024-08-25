from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_85fa5666(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flowing colors to their target regions with specific rules:
    
    1. Colors flow towards their target corners: Sky Blue (8) to top-left, Green (3) to top-right,
       Orange (7) to bottom-left, Magenta (6) to bottom-right.
    2. Colors transform when they interact: Green can become Magenta, Orange can become Green,
       Magenta can become Sky Blue.
    3. Red (2) 2x2 blocks remain unchanged and block color flow.
    4. Colors flow around obstacles and can coexist in adjacent cells.
    5. The final grid aims for balance and often symmetry.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    red_blocks = identify_red_blocks(output_grid)
    colored_cells = identify_colored_cells(output_grid, red_blocks)
    
    for color in [8, 3, 7, 6]:
        flow_color(output_grid, colored_cells, color, red_blocks)
    
    balance_grid(output_grid, red_blocks)
    return output_grid

def identify_red_blocks(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    red_blocks = set()
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 2 for dr in range(2) for dc in range(2)):
                red_blocks.update((r+dr, c+dc) for dr in range(2) for dc in range(2))
    return red_blocks

def identify_colored_cells(grid: ColoredGrid, red_blocks: Set[Tuple[int, int]]) -> Dict[int, List[Tuple[int, int]]]:
    colored_cells = {8: [], 3: [], 7: [], 6: []}
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            if color in colored_cells and (r, c) not in red_blocks:
                colored_cells[color].append((r, c))
    return colored_cells

def flow_color(grid: ColoredGrid, colored_cells: Dict[int, List[Tuple[int, int]]], color: int, red_blocks: Set[Tuple[int, int]]):
    target = get_target_corner(color, grid.get_dimensions())
    for r, c in colored_cells[color]:
        flow_from_cell(grid, r, c, color, target, red_blocks)

def get_target_corner(color: int, dimensions: Tuple[int, int]) -> Tuple[int, int]:
    rows, cols = dimensions
    return {
        8: (0, 0),           # Sky Blue to top-left
        3: (0, cols - 1),    # Green to top-right
        7: (rows - 1, 0),    # Orange to bottom-left
        6: (rows - 1, cols - 1)  # Magenta to bottom-right
    }[color]

def flow_from_cell(grid: ColoredGrid, r: int, c: int, color: int, target: Tuple[int, int], red_blocks: Set[Tuple[int, int]]):
    queue = deque([(r, c)])
    visited = set()
    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if is_valid_cell(nr, nc, grid) and (nr, nc) not in red_blocks:
                cell_color = grid.get_cell(nr, nc)
                if cell_color == 0 or should_transform(color, cell_color):
                    new_color = transform_color(color, cell_color)
                    grid.set_cell(nr, nc, new_color)
                    queue.append((nr, nc))
                elif cell_color == color:
                    queue.append((nr, nc))

def is_valid_cell(row: int, col: int, grid: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols

def should_transform(color1: int, color2: int) -> bool:
    transformations = {(3, 6), (7, 3), (6, 8)}
    return (color1, color2) in transformations

def transform_color(color1: int, color2: int) -> int:
    if (color1, color2) == (3, 6) or color2 == 6:
        return 6
    elif (color1, color2) == (7, 3) or color2 == 3:
        return 3
    elif (color1, color2) == (6, 8) or color2 == 8:
        return 8
    return color1

def balance_grid(grid: ColoredGrid, red_blocks: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in red_blocks:
                balance_cell(grid, r, c, red_blocks)

def balance_cell(grid: ColoredGrid, r: int, c: int, red_blocks: Set[Tuple[int, int]]):
    color_counts = {}
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if is_valid_cell(nr, nc, grid) and (nr, nc) not in red_blocks:
                color = grid.get_cell(nr, nc)
                if color != 0:
                    color_counts[color] = color_counts.get(color, 0) + 1
    
    if color_counts:
        most_common_color = max(color_counts, key=color_counts.get)
        grid.set_cell(r, c, most_common_color)
