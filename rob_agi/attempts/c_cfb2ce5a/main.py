from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_cfb2ce5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid expansion challenge by expanding each color while preserving the original pattern
    and maintaining a black border.

    The algorithm works as follows:
    1. Initialize a copy of the input grid and identify all unique non-zero colors.
    2. For each color, identify its original positions as "seed" positions.
    3. Expand colors from their seed positions, maintaining relative positions and patterns.
    4. Repeat the expansion process until no more pattern-preserving expansions are possible.
    5. Fill any remaining black cells with the nearest non-black neighbor's color.
    6. Ensure the outermost border remains black.
    7. Return the transformed grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid after applying the expansion algorithm.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def get_unique_colors() -> List[int]:
        return sorted(set(grid.values[r][c] for r in range(rows) for c in range(cols) if grid.values[r][c] != 0))
    
    def get_seed_positions(color: int) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == color]
    
    def can_expand(r: int, c: int, color: int) -> bool:
        if r < 1 or r >= rows - 1 or c < 1 or c >= cols - 1:
            return False
        return grid.values[r][c] == 0 and not breaks_pattern(r, c, color)
    
    def breaks_pattern(r: int, c: int, color: int) -> bool:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid.values[nr][nc] != 0 and grid.values[nr][nc] != color:
                    opposite_r, opposite_c = nr + dr, nc + dc
                    if 0 <= opposite_r < rows and 0 <= opposite_c < cols:
                        if grid.values[opposite_r][opposite_c] == grid.values[nr][nc]:
                            return True
        return False
    
    def expand_color(color: int):
        seeds = get_seed_positions(color)
        new_seeds = []
        for r, c in seeds:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if can_expand(nr, nc, color):
                    grid.values[nr][nc] = color
                    new_seeds.append((nr, nc))
        return new_seeds
    
    def fill_remaining_zeros():
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid.values[r][c] == 0:
                    neighbors = [grid.values[r+dr][c+dc] for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr != 0 or dc != 0)]
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    if non_zero_neighbors:
                        grid.values[r][c] = max(set(non_zero_neighbors), key=non_zero_neighbors.count)
    
    def maintain_border():
        for r in range(rows):
            grid.values[r][0] = grid.values[r][-1] = 0
        for c in range(cols):
            grid.values[0][c] = grid.values[-1][c] = 0
    
    colors = get_unique_colors()
    changed = True
    while changed:
        changed = False
        for color in colors:
            new_seeds = expand_color(color)
            if new_seeds:
                changed = True
    
    fill_remaining_zeros()
    maintain_border()
    return grid
