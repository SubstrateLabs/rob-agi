from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_9c56f360(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving connected components of green (3) squares as far left and up as possible
    while maintaining contact with at least one sky blue (8) square (including diagonally)
    and avoiding overlap with other green squares.

    1. Identifies connected components of green squares in the grid.
    2. Processes each component from left to right, top to bottom.
    3. For each component, finds the best new position (leftmost, then topmost) for the entire component.
    4. Moves the component to the new position if different from the original.
    5. If a component cannot be moved as a whole, it is split into smaller components and the process is repeated.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with green square components moved.
    """
    grid = input_grid.deep_copy()
    components = find_connected_components(grid)
    components.sort(key=lambda c: (min(col for _, col in c), min(row for row, _ in c)))
    
    for component in components:
        move_component(grid, component)
    
    return grid

def find_connected_components(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    components = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                component = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 3:
                        visited.add((curr_r, curr_c))
                        component.append((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if is_valid_position(grid, nr, nc):
                                stack.append((nr, nc))
                components.append(component)
    
    return components

def is_valid_position(grid: ColoredGrid, row: int, col: int) -> bool:
    rows, cols = grid.get_dimensions()
    return 0 <= row < rows and 0 <= col < cols

def is_adjacent_to_sky_blue(grid: ColoredGrid, positions: List[Tuple[int, int]]) -> bool:
    for row, col in positions:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if is_valid_position(grid, nr, nc) and grid.get_cell(nr, nc) == 8:
                    return True
    return False

def find_valid_positions(grid: ColoredGrid, component: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    valid_positions = []
    rows, cols = grid.get_dimensions()
    component_set = set(component)
    
    for r in range(rows):
        for c in range(cols):
            is_valid = True
            for dr, dc in enumerate(component):
                nr, nc = r + dr, c + dc
                if not is_valid_position(grid, nr, nc) or (grid.get_cell(nr, nc) == 3 and (nr, nc) not in component_set):
                    is_valid = False
                    break
            if is_valid and is_adjacent_to_sky_blue(grid, [(r + dr, c + dc) for dr, dc in enumerate(component)]):
                valid_positions.append((r, c))
    
    return valid_positions

def move_component(grid: ColoredGrid, component: List[Tuple[int, int]]) -> None:
    valid_positions = find_valid_positions(grid, component)
    if not valid_positions:
        if len(component) > 1:
            # Split the component and try to move each part
            for sub_component in split_component(component):
                move_component(grid, sub_component)
        return
    
    best_position = min(valid_positions, key=lambda p: (p[1], p[0]))  # Leftmost, then topmost
    
    # Move the component
    for i, (row, col) in enumerate(component):
        grid.set_cell(row, col, 0)  # Set original position to black
        new_row, new_col = best_position[0] + i, best_position[1]
        grid.set_cell(new_row, new_col, 3)  # Set new position to green

def split_component(component: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    # Split the component into individual squares
    return [[pos] for pos in component]
