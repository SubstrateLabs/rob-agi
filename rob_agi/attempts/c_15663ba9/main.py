from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying and modifying connected components.
    
    The solution follows these steps:
    1. Find all connected components of non-zero colors.
    2. For each component:
       - Mark outermost corners and endpoints as yellow (4).
       - Identify critical points (internal corners and junctions) as red (2).
    3. Handle special cases like 2x2 squares.
    4. Preserve the original black (0) background.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def find_connected_components() -> List[Set[Tuple[int, int]]]:
        components = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) != 0 and (r, c) not in visited:
                    component = set()
                    stack = [(r, c)]
                    color = output_grid.get_cell(r, c)
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and output_grid.get_cell(curr_r, curr_c) == color:
                            visited.add((curr_r, curr_c))
                            component.add((curr_r, curr_c))
                            stack.extend(get_neighbors(curr_r, curr_c))
                    components.append(component)
        return components

    def is_outermost_corner_or_endpoint(r: int, c: int, component: Set[Tuple[int, int]]) -> bool:
        neighbors = [n for n in get_neighbors(r, c) if n in component]
        return len(neighbors) < 2

    def is_critical_point(r: int, c: int, component: Set[Tuple[int, int]]) -> bool:
        neighbors = [n for n in get_neighbors(r, c) if n in component]
        if len(neighbors) == 2:
            return abs(neighbors[0][0] - neighbors[1][0]) + abs(neighbors[0][1] - neighbors[1][1]) == 2
        return len(neighbors) >= 3

    def is_2x2_square(r: int, c: int) -> bool:
        if r + 1 < rows and c + 1 < cols:
            color = output_grid.get_cell(r, c)
            return all(output_grid.get_cell(r+dr, c+dc) == color for dr, dc in [(0,0), (0,1), (1,0), (1,1)])
        return False

    components = find_connected_components()

    for component in components:
        for r, c in component:
            if is_outermost_corner_or_endpoint(r, c, component):
                output_grid.set_cell(r, c, 4)  # Yellow
            elif is_critical_point(r, c, component):
                output_grid.set_cell(r, c, 2)  # Red

    for r in range(rows):
        for c in range(cols):
            if is_2x2_square(r, c) and output_grid.get_cell(r, c) != 4:
                output_grid.set_cell(r, c, 2)  # Red

    return output_grid
