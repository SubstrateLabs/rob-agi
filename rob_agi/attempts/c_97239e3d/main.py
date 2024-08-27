from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_97239e3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding colored squares based on priority.
    
    The solution follows these steps:
    1. Identify expansion colors and their positions.
    2. Sort colors based on priority (closer to corners have higher priority).
    3. Expand each color using a flood-fill algorithm, preserving the sky blue (8) pattern.
    4. Repeat the expansion process until no changes are made.
    5. Preserve the 17th row and column (index 16) from the original grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def is_expandable(r: int, c: int) -> bool:
        return 0 <= r < 16 and 0 <= c < 16 and output_grid.get_cell(r, c) in [0, 8]
    
    def get_priority(r: int, c: int) -> float:
        return min(r, 15-r, c, 15-c)
    
    def find_expansion_colors() -> List[Tuple[int, List[Tuple[int, int]]]]:
        colors: Dict[int, List[Tuple[int, int]]] = {}
        for r in range(16):
            for c in range(16):
                color = output_grid.get_cell(r, c)
                if color not in [0, 8]:
                    if color not in colors:
                        colors[color] = []
                    colors[color].append((r, c))
        return sorted(colors.items(), key=lambda x: min(get_priority(r, c) for r, c in x[1]))
    
    def expand_color(color: int, start_positions: List[Tuple[int, int]]) -> bool:
        changed = False
        queue = deque(start_positions)
        visited = set(start_positions)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and is_expandable(nr, nc):
                    if output_grid.get_cell(nr, nc) == 8:
                        output_grid.set_cell(nr, nc, color)
                        changed = True
                    elif output_grid.get_cell(nr, nc) == 0:
                        output_grid.set_cell(nr, nc, color)
                        queue.append((nr, nc))
                        changed = True
                    visited.add((nr, nc))
        return changed
    
    changes_made = True
    while changes_made:
        changes_made = False
        for color, positions in find_expansion_colors():
            changes_made |= expand_color(color, positions)
    
    # Preserve the 17th row and column
    for i in range(17):
        output_grid.set_cell(16, i, input_grid.get_cell(16, i))
        output_grid.set_cell(i, 16, input_grid.get_cell(i, 16))
    
    return output_grid
