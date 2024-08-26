from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import itertools

def solve_ce039d91(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid by changing some gray (5) cells to blue (1) based on their structural importance.
    
    The solution works as follows:
    1. Identify all gray (5) cells and create a map of the grid's structure.
    2. Calculate a structural importance score for each gray cell based on adjacency and position.
    3. Identify key structural elements like junction points and shape-defining cells.
    4. Transform cells to blue (1) if their structural importance is below a threshold.
    5. Handle special cases like linear shapes, 2x2 squares, and implicit structures.
    6. Perform consistency checks and fine-tune the transformation.
    
    This approach considers both local and global patterns, allowing for context-dependent 
    transformations while maintaining the overall structure and logic of the original pattern.
    """
    def find_gray_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
        return [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 5]

    def calculate_structural_importance(grid: List[List[int]], cell: Tuple[int, int]) -> int:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        score = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 5:
                    score += 1
        return score

    def is_junction_point(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        gray_neighbors = 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 5:
                gray_neighbors += 1
        return gray_neighbors > 2

    def is_linear_shape(cells: List[Tuple[int, int]]) -> bool:
        if len(cells) <= 3:
            return True
        r_coords, c_coords = zip(*cells)
        return len(set(r_coords)) == 1 or len(set(c_coords)) == 1

    def is_2x2_square(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        if r + 1 < rows and c + 1 < cols:
            return all(grid[r+dr][c+dc] == 5 for dr in [0, 1] for dc in [0, 1])
        return False

    new_grid = [row[:] for row in input_grid.values]
    gray_cells = find_gray_cells(new_grid)
    
    # Calculate structural importance for each gray cell
    cell_scores = {cell: calculate_structural_importance(new_grid, cell) for cell in gray_cells}
    
    # Identify key structural elements
    junction_points = [cell for cell in gray_cells if is_junction_point(new_grid, cell)]
    linear_shapes = [shape for shape in [list(group) for _, group in itertools.groupby(sorted(gray_cells))] if is_linear_shape(shape)]
    
    # Transform cells based on structural importance
    threshold = sum(cell_scores.values()) / len(cell_scores) if cell_scores else 0
    for cell in gray_cells:
        if cell in junction_points or is_2x2_square(new_grid, cell):
            continue  # Keep these cells gray
        if cell_scores[cell] < threshold or any(cell in shape for shape in linear_shapes):
            new_grid[cell[0]][cell[1]] = 1  # Change to blue
    
    # Consistency check and fine-tuning
    for r in range(len(new_grid)):
        for c in range(len(new_grid[0])):
            if new_grid[r][c] == 1:
                # If a blue cell is surrounded by gray, change it back to gray
                if all(0 <= r+dr < len(new_grid) and 0 <= c+dc < len(new_grid[0]) and new_grid[r+dr][c+dc] == 5 
                       for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    new_grid[r][c] = 5

    return ColoredGrid(values=new_grid)
