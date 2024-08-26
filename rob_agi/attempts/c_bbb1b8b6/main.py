from rob_agi.colored_grid import ColoredGrid
from typing import List, Set, Tuple
from collections import deque, Counter

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input 4x9 grid into a 4x4 grid based on the following rules:
    
    1. Extract the left half (first 4 columns) of the input grid.
    2. Expand the shape from the left half into the right half where there's a connection.
    3. If no expansion occurs, use only the left half.
    4. Fill any remaining empty spaces with the most common non-zero color.
    5. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    left_half = extract_left_half(input_grid)
    right_half = extract_right_half(input_grid)
    
    result_grid = expand_shape(left_half, right_half)
    fill_empty_spaces(result_grid)
    
    return ColoredGrid(values=result_grid)

def extract_left_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[:4] for row in input_grid.values]

def extract_right_half(input_grid: ColoredGrid) -> List[List[int]]:
    return [row[5:] for row in input_grid.values]

def expand_shape(left_half: List[List[int]], right_half: List[List[int]]) -> List[List[int]]:
    result = [row[:] for row in left_half]
    shape_set = {(r, c) for r in range(4) for c in range(4) if left_half[r][c] != 0}
    initial_size = len(shape_set)
    
    queue = deque((r, c) for r in range(4) for c in range(4) if right_half[r][c] != 0)
    checked = set()
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in checked:
            continue
        checked.add((r, c))
        
        if is_adjacent(r, c, shape_set):
            result[r][c] = right_half[r][c]
            shape_set.add((r, c))
            for nr, nc in get_neighbors(r, c):
                if 0 <= nr < 4 and 0 <= nc < 4 and right_half[nr][nc] != 0:
                    queue.append((nr, nc))
    
    if len(shape_set) == initial_size:
        return left_half
    return result

def is_adjacent(r: int, c: int, shape_set: Set[Tuple[int, int]]) -> bool:
    for nr, nc in get_neighbors(r, c):
        if (nr, nc) in shape_set:
            return True
    return False

def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    return [(r-1, c), (r+1, c), (r, c-1), (r, 3-c)]

def fill_empty_spaces(grid: List[List[int]]) -> None:
    colors = [cell for row in grid for cell in row if cell != 0]
    main_color = Counter(colors).most_common(1)[0][0] if colors else 1
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0:
                grid[r][c] = main_color
