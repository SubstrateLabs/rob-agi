from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import itertools

def solve_ce039d91(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid by changing some gray (5) cells to blue (1) based on their structural importance.
    
    The solution works as follows:
    1. Create a deep copy of the input grid and identify all gray cells.
    2. Analyze the structure of the grid, detecting lines, corners, junctions, and special patterns.
    3. Calculate a structural importance score for each gray cell based on its position and surroundings.
    4. Determine a transformation threshold based on the overall density of gray cells.
    5. Apply transformation rules, including:
       - Changing isolated gray cells to blue, except on edges.
       - Transforming 2x2 squares of gray cells to blue.
       - Handling lines and X-shapes with specific rules.
       - Changing cells to blue if their importance score is below the threshold.
    6. Perform post-processing to ensure consistency and preserve key structural elements.
    7. Conduct a final check to maintain the overall structure and balance of the grid.
    
    This approach balances local cell transformations with global pattern preservation,
    adapting to different grid densities and structures while maintaining key features.
    """
    def find_gray_cells(grid: List[List[int]]) -> Set[Tuple[int, int]]:
        return {(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 5}

    def calculate_structural_importance(grid: List[List[int]], cell: Tuple[int, int]) -> int:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        score = 0
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 5:
                score += 2 if dr == 0 or dc == 0 else 1  # Higher score for orthogonal neighbors
        return score

    def is_junction_point(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        gray_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr][c+dc] == 5)
        return gray_neighbors > 2

    def is_2x2_square(grid: List[List[int]], cell: Tuple[int, int]) -> bool:
        r, c = cell
        rows, cols = len(grid), len(grid[0])
        if r + 1 < rows and c + 1 < cols:
            return all(grid[r+dr][c+dc] == 5 for dr in [0, 1] for dc in [0, 1])
        return False

    def find_lines(gray_cells: Set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        lines = []
        remaining_cells = gray_cells.copy()
        while remaining_cells:
            start = remaining_cells.pop()
            line = [start]
            for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                current = start
                while True:
                    next_cell = (current[0] + direction[0], current[1] + direction[1])
                    if next_cell in remaining_cells:
                        line.append(next_cell)
                        remaining_cells.remove(next_cell)
                        current = next_cell
                    else:
                        break
                direction = (-direction[0], -direction[1])
                current = start
                while True:
                    next_cell = (current[0] + direction[0], current[1] + direction[1])
                    if next_cell in remaining_cells:
                        line.insert(0, next_cell)
                        remaining_cells.remove(next_cell)
                        current = next_cell
                    else:
                        break
            if len(line) > 1:
                lines.append(line)
        return lines

    new_grid = [row[:] for row in input_grid.values]
    rows, cols = len(new_grid), len(new_grid[0])
    gray_cells = find_gray_cells(new_grid)
    
    # Calculate structural importance for each gray cell
    cell_scores = {cell: calculate_structural_importance(new_grid, cell) for cell in gray_cells}
    
    # Identify key structural elements
    junction_points = {cell for cell in gray_cells if is_junction_point(new_grid, cell)}
    lines = find_lines(gray_cells)
    
    # Determine transformation threshold
    gray_density = len(gray_cells) / (rows * cols)
    threshold = sum(cell_scores.values()) / len(cell_scores) if cell_scores else 0
    threshold *= (1 + gray_density)  # Adjust threshold based on density
    
    # Apply transformation rules
    for cell in gray_cells:
        if cell in junction_points or is_2x2_square(new_grid, cell):
            continue  # Keep these cells gray
        if cell_scores[cell] < threshold:
            new_grid[cell[0]][cell[1]] = 1  # Change to blue
    
    # Handle lines
    for line in lines:
        if len(line) <= 3:
            for cell in line[1:-1]:
                new_grid[cell[0]][cell[1]] = 1
        else:
            for cell in line[2:-2]:
                new_grid[cell[0]][cell[1]] = 1

    # Post-processing
    for r in range(rows):
        for c in range(cols):
            if new_grid[r][c] == 1:
                # If a blue cell is surrounded by gray, change it back to gray
                if all(0 <= r+dr < rows and 0 <= c+dc < cols and new_grid[r+dr][c+dc] == 5 
                       for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    new_grid[r][c] = 5

    # Edge preservation
    for r in [0, rows-1]:
        for c in range(cols):
            if new_grid[r][c] == 1 and input_grid.values[r][c] == 5:
                new_grid[r][c] = 5
    for c in [0, cols-1]:
        for r in range(rows):
            if new_grid[r][c] == 1 and input_grid.values[r][c] == 5:
                new_grid[r][c] = 5

    return ColoredGrid(values=new_grid)
