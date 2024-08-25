from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Set

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing green (3) color and extending adjacent patterns
    while preserving the main structure and symmetry. The solution involves:
    1. Preserving the border pattern exactly.
    2. Replacing green cells with colors that maintain symmetry and extend existing patterns.
    3. Ensuring perfect 180-degree rotational symmetry in the final grid.
    4. Performing multiple passes to catch and correct any inconsistencies.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_symmetrical_cell(row: int, col: int) -> Tuple[int, int]:
        return rows - 1 - row, cols - 1 - col
    
    def get_neighborhood(grid: ColoredGrid, row: int, col: int, size: int = 1) -> List[int]:
        neighbors = []
        for r in range(row - size, row + size + 1):
            for c in range(col - size, col + size + 1):
                if 0 <= r < rows and 0 <= c < cols and (r != row or c != col):
                    neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_replacement_color(neighbors: List[int]) -> int:
        return Counter([n for n in neighbors if n != 3]).most_common(1)[0][0]
    
    def is_border(row: int, col: int) -> bool:
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1
    
    def apply_symmetrical(grid: ColoredGrid, row: int, col: int, color: int):
        sym_row, sym_col = get_symmetrical_cell(row, col)
        grid.values[row][col] = color
        grid.values[sym_row][sym_col] = color
    
    # Preserve border
    for i in range(cols):
        output_grid.values[0][i] = input_grid.values[0][i]
        output_grid.values[-1][i] = input_grid.values[-1][i]
    for i in range(1, rows - 1):
        output_grid.values[i][0] = input_grid.values[i][0]
        output_grid.values[i][-1] = input_grid.values[i][-1]
    
    # Replace green cells
    green_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 3]
    for r, c in green_cells:
        if not is_border(r, c):
            sym_r, sym_c = get_symmetrical_cell(r, c)
            sym_color = output_grid.values[sym_r][sym_c]
            if sym_color != 3:
                apply_symmetrical(output_grid, r, c, sym_color)
            else:
                neighbors = get_neighborhood(output_grid, r, c, size=2) + get_neighborhood(output_grid, sym_r, sym_c, size=2)
                new_color = get_replacement_color(neighbors)
                apply_symmetrical(output_grid, r, c, new_color)
    
    # Symmetry correction
    for r in range(rows // 2 + 1):
        for c in range(cols):
            sym_r, sym_c = get_symmetrical_cell(r, c)
            if output_grid.values[r][c] != output_grid.values[sym_r][sym_c]:
                neighbors = get_neighborhood(output_grid, r, c) + get_neighborhood(output_grid, sym_r, sym_c)
                new_color = get_replacement_color(neighbors)
                apply_symmetrical(output_grid, r, c, new_color)
    
    return output_grid
