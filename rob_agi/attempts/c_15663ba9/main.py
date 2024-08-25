from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set, Dict
from collections import deque

def solve_15663ba9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by analyzing shape structures and color coding.
    
    The solution follows these steps:
    1. Identify connected components of non-zero colors.
    2. For each component:
       - Analyze the shape structure using a skeleton approach.
       - Mark extremities (endpoints, single-cell protrusions) as yellow (4).
       - Identify structural transition points (corners, junctions) as red (2).
    3. Handle special cases like 2x2 squares and single-cell shapes.
    4. Preserve the original colors for non-marked cells and the black (0) background.

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

    def analyze_shape(component: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        color_map = {}
        skeleton = find_skeleton(component)
        
        # Mark extremities as yellow
        for cell in component:
            if len([n for n in get_neighbors(*cell) if n in component]) == 1:
                color_map[cell] = 4  # Yellow
        
        # Mark structural points as red
        for cell in skeleton:
            neighbors = [n for n in get_neighbors(*cell) if n in skeleton]
            if len(neighbors) != 2 or is_corner(*cell, neighbors):
                color_map[cell] = 2  # Red
        
        return color_map

    def find_skeleton(component: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        skeleton = set(component)
        border = set()
        for cell in component:
            if any(n not in component for n in get_neighbors(*cell)):
                border.add(cell)
        
        while border:
            new_border = set()
            for cell in border:
                neighbors = [n for n in get_neighbors(*cell) if n in skeleton]
                if len(neighbors) <= 1:
                    skeleton.remove(cell)
                    new_border.update(neighbors)
            border = new_border
        
        return skeleton

    def is_corner(r: int, c: int, neighbors: List[Tuple[int, int]]) -> bool:
        if len(neighbors) != 2:
            return False
        return abs(neighbors[0][0] - neighbors[1][0]) + abs(neighbors[0][1] - neighbors[1][1]) == 2

    def is_2x2_square(r: int, c: int) -> bool:
        if r + 1 < rows and c + 1 < cols:
            color = output_grid.get_cell(r, c)
            return all(output_grid.get_cell(r+dr, c+dc) == color for dr, dc in [(0,0), (0,1), (1,0), (1,1)])
        return False

    components = find_connected_components()

    for component in components:
        if len(component) == 1:
            r, c = next(iter(component))
            output_grid.set_cell(r, c, 4)  # Single cell becomes yellow
        else:
            color_map = analyze_shape(component)
            for cell, color in color_map.items():
                output_grid.set_cell(*cell, color)

    # Handle 2x2 squares
    for r in range(rows):
        for c in range(cols):
            if is_2x2_square(r, c) and output_grid.get_cell(r, c) != 4:
                for dr, dc in [(0,0), (0,1), (1,0), (1,1)]:
                    output_grid.set_cell(r+dr, c+dc, 2)  # Red

    return output_grid
