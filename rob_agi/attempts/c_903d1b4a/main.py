from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Set

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing green (3) color and extending adjacent patterns
    while preserving the main structure and symmetry. The solution involves:
    1. Preserving the border pattern exactly.
    2. Identifying and maintaining (or completing) the central pattern.
    3. Replacing green cells by extending or completing existing patterns.
    4. Ensuring symmetry is maintained throughout the process.
    5. Performing multiple passes to catch and correct any inconsistencies.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_central_area(grid: ColoredGrid) -> List[List[int]]:
        center_size = min(rows, cols) // 2
        start_row, start_col = (rows - center_size) // 2, (cols - center_size) // 2
        return grid.extract_subgrid(start_row, start_col, center_size, center_size).values
    
    def get_neighborhood(grid: ColoredGrid, row: int, col: int, size: int = 1) -> List[int]:
        neighbors = []
        for r in range(row - size, row + size + 1):
            for c in range(col - size, col + size + 1):
                if 0 <= r < rows and 0 <= c < cols and (r != row or c != col):
                    neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_replacement_color(neighbors: List[int]) -> int:
        return Counter(neighbors).most_common(1)[0][0]
    
    def is_border(row: int, col: int) -> bool:
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1
    
    def apply_symmetrical(grid: ColoredGrid, row: int, col: int, color: int):
        grid.values[row][col] = color
        grid.values[row][cols - 1 - col] = color
        grid.values[rows - 1 - row][col] = color
        grid.values[rows - 1 - row][cols - 1 - col] = color
    
    # Preserve border
    for i in range(cols):
        output_grid.values[0][i] = input_grid.values[0][i]
        output_grid.values[-1][i] = input_grid.values[-1][i]
    for i in range(1, rows - 1):
        output_grid.values[i][0] = input_grid.values[i][0]
        output_grid.values[i][-1] = input_grid.values[i][-1]
    
    # Process central area
    central_area = get_central_area(input_grid)
    central_pattern = [color for row in central_area for color in row if color != 3]
    
    # Replace green cells
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 3:
                if not is_border(r, c):
                    neighbors = get_neighborhood(output_grid, r, c, size=2)
                    new_color = get_replacement_color([color for color in neighbors if color != 3])
                    apply_symmetrical(output_grid, r, c, new_color)
    
    # Pattern extension and symmetry correction
    for _ in range(2):  # Two passes for better consistency
        for r in range(rows):
            for c in range(cols):
                if not is_border(r, c):
                    neighbors = get_neighborhood(output_grid, r, c)
                    most_common = Counter(neighbors).most_common(2)
                    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                        new_color = get_replacement_color(central_pattern)
                    else:
                        new_color = most_common[0][0]
                    apply_symmetrical(output_grid, r, c, new_color)
    
    return output_grid
