from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing green (3) color and extending adjacent patterns
    while preserving the main structure and symmetry. The solution involves:
    1. Preserving the border pattern exactly.
    2. Replacing green cells with colors that maintain symmetry and extend existing patterns.
    3. Ensuring perfect 180-degree rotational symmetry in the final grid.
    4. Performing multiple passes to refine the solution and ensure consistency.
    5. Considering both local and global patterns, including specific shapes and quadrant consistency.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_symmetrical_cell(row: int, col: int) -> Tuple[int, int]:
        return rows - 1 - row, cols - 1 - col
    
    def get_neighborhood(grid: ColoredGrid, row: int, col: int, size: int = 2) -> List[int]:
        neighbors = []
        for r in range(row - size, row + size + 1):
            for c in range(col - size, col + size + 1):
                if 0 <= r < rows and 0 <= c < cols and (r != row or c != col):
                    neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_replacement_color(neighbors: List[int], patterns: Dict[str, List[int]]) -> int:
        color_count = Counter([n for n in neighbors if n != 3])
        for pattern, colors in patterns.items():
            if all(color in neighbors for color in colors):
                return max(set(colors), key=colors.count)
        return color_count.most_common(1)[0][0] if color_count else 1  # Default to blue if no other colors
    
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
    
    # Define common patterns
    patterns = {
        "diamond": [7, 8, 5],
        "cross": [1, 6, 7],
        "checkerboard": [1, 6, 4, 7]
    }
    
    # First pass - Replace green cells
    green_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 3]
    for r, c in green_cells:
        if not is_border(r, c):
            sym_r, sym_c = get_symmetrical_cell(r, c)
            sym_color = output_grid.values[sym_r][sym_c]
            if sym_color != 3:
                apply_symmetrical(output_grid, r, c, sym_color)
            else:
                neighbors = get_neighborhood(output_grid, r, c) + get_neighborhood(output_grid, sym_r, sym_c)
                new_color = get_replacement_color(neighbors, patterns)
                apply_symmetrical(output_grid, r, c, new_color)
    
    # Second pass - Pattern continuity
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if not is_border(r, c):
                neighbors = get_neighborhood(output_grid, r, c, size=1)
                current_color = output_grid.values[r][c]
                if current_color not in neighbors:
                    new_color = get_replacement_color(neighbors, patterns)
                    apply_symmetrical(output_grid, r, c, new_color)
    
    # Third pass - Quadrant consistency
    quadrant_size = rows // 2
    for qr in range(2):
        for qc in range(2):
            quadrant = [output_grid.values[r][c] for r in range(qr*quadrant_size, (qr+1)*quadrant_size)
                        for c in range(qc*quadrant_size, (qc+1)*quadrant_size)]
            dominant_color = Counter(quadrant).most_common(1)[0][0]
            for r in range(qr*quadrant_size, (qr+1)*quadrant_size):
                for c in range(qc*quadrant_size, (qc+1)*quadrant_size):
                    if not is_border(r, c) and output_grid.values[r][c] != dominant_color:
                        neighbors = get_neighborhood(output_grid, r, c, size=1)
                        if dominant_color in neighbors:
                            apply_symmetrical(output_grid, r, c, dominant_color)
    
    # Fourth pass - Border transition
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if r == 1 or r == rows - 2 or c == 1 or c == cols - 2:
                neighbors = get_neighborhood(output_grid, r, c, size=1)
                border_colors = [output_grid.values[nr][nc] for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] if is_border(nr, nc)]
                if output_grid.values[r][c] not in border_colors and any(color in neighbors for color in border_colors):
                    new_color = Counter(border_colors).most_common(1)[0][0]
                    apply_symmetrical(output_grid, r, c, new_color)
    
    # Final symmetry check
    for r in range(rows // 2):
        for c in range(cols):
            sym_r, sym_c = get_symmetrical_cell(r, c)
            if output_grid.values[r][c] != output_grid.values[sym_r][sym_c]:
                neighbors = get_neighborhood(output_grid, r, c) + get_neighborhood(output_grid, sym_r, sym_c)
                new_color = get_replacement_color(neighbors, patterns)
                apply_symmetrical(output_grid, r, c, new_color)
    
    return output_grid
