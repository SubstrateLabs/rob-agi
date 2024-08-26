from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_ac0c5833(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding red (2) regions around yellow (4) cells,
    replicating complex patterns, and ensuring connectivity. The function:
    1. Identifies yellow cells and existing red structures.
    2. Expands 3x3 red regions around yellow cells, leaving one corner empty.
    3. Replicates recognized red patterns relative to yellow cells.
    4. Connects nearby red expansions and structures.
    5. Applies consistent rules for red cell placement.
    6. Resolves overlaps, prioritizing existing structures.
    7. Ensures overall connectivity and removes isolated red cells.
    8. Handles edge cases and different grid sizes.
    """
    grid = input_grid.deep_copy()
    yellow_cells = [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 4]
    patterns = recognize_patterns(grid)
    
    for row, col in yellow_cells:
        expand_around_yellow(grid, row, col)
        replicate_pattern(grid, row, col, patterns)
    
    connect_expansions(grid)
    ensure_connectivity(grid)
    remove_isolated_reds(grid)
    
    return grid

def recognize_patterns(grid: ColoredGrid) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    patterns = {}
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 4:
                pattern = []
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                            if grid.values[nr][nc] == 2:
                                pattern.append((dr, dc))
                if pattern:
                    patterns[(r, c)] = pattern
    return patterns

def expand_around_yellow(grid: ColoredGrid, row: int, col: int):
    corners = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
    empty_corner = choose_empty_corner(grid, corners)
    
    for i in range(max(0, row-1), min(grid.num_rows, row+2)):
        for j in range(max(0, col-1), min(grid.num_cols, col+2)):
            if (i, j) != (row, col) and (i, j) != empty_corner:
                if grid.values[i][j] == 0:  # Only change if it's currently black
                    grid.values[i][j] = 2  # Change to red

def choose_empty_corner(grid: ColoredGrid, corners: List[Tuple[int, int]]) -> Tuple[int, int]:
    for corner in corners:
        if 0 <= corner[0] < grid.num_rows and 0 <= corner[1] < grid.num_cols:
            if grid.values[corner[0]][corner[1]] == 0:
                return corner
    return corners[0]  # Default to top-left if no empty corner found

def replicate_pattern(grid: ColoredGrid, row: int, col: int, patterns: Dict[Tuple[int, int], List[Tuple[int, int]]]):
    for (pr, pc), pattern in patterns.items():
        if (row, col) != (pr, pc):  # Don't replicate on the original yellow cell
            for dr, dc in pattern:
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                    if grid.values[nr][nc] == 0:
                        grid.values[nr][nc] = 2

def connect_expansions(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2:
                connect_neighbors(grid, r, c)

def connect_neighbors(grid: ColoredGrid, row: int, col: int):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            if grid.values[nr][nc] == 0:
                diagonal_reds = sum(1 for ddr, ddc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                                    if 0 <= nr+ddr < grid.num_rows and 0 <= nc+ddc < grid.num_cols
                                    and grid.values[nr+ddr][nc+ddc] == 2)
                if diagonal_reds >= 2:
                    grid.values[nr][nc] = 2

def ensure_connectivity(grid: ColoredGrid):
    visited = set()
    largest_component = set()
    
    def dfs(r, c):
        stack = [(r, c)]
        component = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and grid.values[r][c] == 2:
                visited.add((r, c))
                component.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
                        stack.append((nr, nc))
        return component
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and (r, c) not in visited:
                component = dfs(r, c)
                if len(component) > len(largest_component):
                    largest_component = component
    
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and (r, c) not in largest_component:
                grid.values[r][c] = 0

def remove_isolated_reds(grid: ColoredGrid):
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] == 2 and is_isolated(grid, r, c):
                grid.values[r][c] = 0

def is_isolated(grid: ColoredGrid, row: int, col: int) -> bool:
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid.num_rows and 0 <= nc < grid.num_cols:
            if grid.values[nr][nc] in [2, 4]:
                return False
    return True
