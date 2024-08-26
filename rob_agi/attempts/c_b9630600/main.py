from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_b9630600(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the b9630600 challenge by maximally filling the space occupied by and between original green shapes.

    The solution follows these steps:
    1. Identify the outermost boundaries of green shapes
    2. Fill the interior space with green
    3. Preserve characteristic "holes" within shapes
    4. Ensure connectivity of all green cells
    5. Maintain symmetry and balance of the structure
    6. Clean up and finalize the result

    This approach creates a fully connected green structure that maintains the overall
    "silhouette" and key features of the original configuration.
    """
    output_grid = input_grid.deep_copy()
    boundaries = find_boundaries(output_grid)
    fill_interior(output_grid, boundaries)
    preserve_holes(output_grid, input_grid)
    ensure_connectivity(output_grid)
    maintain_symmetry(output_grid)
    clean_up(output_grid)
    
    return output_grid

def find_boundaries(grid: ColoredGrid) -> Set[Tuple[int, int]]:
    boundaries = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols and grid.get_cell(nr, nc) == 0:
                        boundaries.add((r, c))
                        break
    return boundaries

def fill_interior(grid: ColoredGrid, boundaries: Set[Tuple[int, int]]):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if (r, c) not in boundaries and grid.get_cell(r, c) == 0:
                if is_inside_boundary(grid, r, c, boundaries):
                    grid.set_cell(r, c, 3)

def is_inside_boundary(grid: ColoredGrid, r: int, c: int, boundaries: Set[Tuple[int, int]]) -> bool:
    crossings = 0
    for i in range(c, grid.num_cols):
        if (r, i) in boundaries:
            crossings += 1
    return crossings % 2 == 1

def preserve_holes(output_grid: ColoredGrid, input_grid: ColoredGrid):
    for r in range(input_grid.num_rows):
        for c in range(input_grid.num_cols):
            if input_grid.get_cell(r, c) == 0 and output_grid.get_cell(r, c) == 3:
                if is_characteristic_hole(input_grid, r, c):
                    flood_fill_hole(output_grid, r, c)

def is_characteristic_hole(grid: ColoredGrid, r: int, c: int) -> bool:
    surrounding_green = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
    return surrounding_green >= 3

def flood_fill_hole(grid: ColoredGrid, r: int, c: int):
    stack = [(r, c)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.get_cell(r, c) == 3:
            grid.set_cell(r, c, 0)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                stack.append((r + dr, c + dc))

def ensure_connectivity(grid: ColoredGrid):
    components = find_connected_components(grid)
    if len(components) > 1:
        main_component = max(components, key=len)
        for component in components:
            if component != main_component:
                connect_components(grid, main_component, component)

def find_connected_components(grid: ColoredGrid) -> List[Set[Tuple[int, int]]]:
    components = []
    visited = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3 and (r, c) not in visited:
                component = set()
                dfs(grid, r, c, component, visited)
                components.append(component)
    return components

def dfs(grid: ColoredGrid, r: int, c: int, component: Set[Tuple[int, int]], visited: Set[Tuple[int, int]]):
    if not (0 <= r < grid.num_rows and 0 <= c < grid.num_cols) or grid.get_cell(r, c) != 3 or (r, c) in visited:
        return
    visited.add((r, c))
    component.add((r, c))
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        dfs(grid, r + dr, c + dc, component, visited)

def connect_components(grid: ColoredGrid, comp1: Set[Tuple[int, int]], comp2: Set[Tuple[int, int]]):
    min_distance = float('inf')
    connection = None
    for r1, c1 in comp1:
        for r2, c2 in comp2:
            distance = abs(r1 - r2) + abs(c1 - c2)
            if distance < min_distance:
                min_distance = distance
                connection = ((r1, c1), (r2, c2))
    if connection:
        r1, c1 = connection[0]
        r2, c2 = connection[1]
        while (r1, c1) != (r2, c2):
            grid.set_cell(r1, c1, 3)
            if r1 < r2:
                r1 += 1
            elif r1 > r2:
                r1 -= 1
            elif c1 < c2:
                c1 += 1
            elif c1 > c2:
                c1 -= 1

def maintain_symmetry(grid: ColoredGrid):
    # Vertical symmetry
    for c in range(grid.num_cols // 2):
        for r in range(grid.num_rows):
            if grid.get_cell(r, c) != grid.get_cell(r, grid.num_cols - 1 - c):
                if grid.get_cell(r, c) == 3:
                    grid.set_cell(r, grid.num_cols - 1 - c, 3)
                else:
                    grid.set_cell(r, c, 3)

    # Horizontal symmetry
    for r in range(grid.num_rows // 2):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) != grid.get_cell(grid.num_rows - 1 - r, c):
                if grid.get_cell(r, c) == 3:
                    grid.set_cell(grid.num_rows - 1 - r, c, 3)
                else:
                    grid.set_cell(r, c, 3)

def clean_up(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) == 3:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < grid.num_rows and 0 <= c + dc < grid.num_cols and grid.get_cell(r + dr, c + dc) == 3)
                if neighbors <= 1:
                    grid.set_cell(r, c, 0)
