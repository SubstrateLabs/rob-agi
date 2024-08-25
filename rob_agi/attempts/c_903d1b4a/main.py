from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Set

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Simplify the pattern in the input grid by removing green (3) color and extending adjacent patterns
    while preserving the main structure and symmetry. The solution involves:
    1. Analyzing the grid to identify green cells, border pattern, and central structures.
    2. Removing green cells and replacing them with colors that extend adjacent patterns.
    3. Applying changes symmetrically to maintain the overall structure.
    4. Preserving border patterns and essential structures.
    5. Performing final passes to ensure consistency and symmetry.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def get_color_frequency(grid: ColoredGrid) -> Counter:
        return Counter(color for row in grid.values for color in row)
    
    def get_border_pattern(grid: ColoredGrid) -> List[int]:
        return (grid.values[0] + grid.values[-1] + 
                [row[0] for row in grid.values[1:-1]] + 
                [row[-1] for row in grid.values[1:-1]])
    
    def get_central_area(grid: ColoredGrid) -> List[List[int]]:
        rows, cols = grid.get_dimensions()
        center_size = min(rows // 2, cols // 2)
        start_row, start_col = (rows - center_size) // 2, (cols - center_size) // 2
        return grid.extract_subgrid(start_row, start_col, center_size, center_size).values
    
    def get_neighborhood(grid: ColoredGrid, row: int, col: int) -> List[int]:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_replacement_color(neighbors: List[int], essential_colors: Set[int]) -> int:
        return Counter([c for c in neighbors if c in essential_colors]).most_common(1)[0][0]
    
    def is_border(row: int, col: int) -> bool:
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1
    
    def apply_symmetrical(grid: ColoredGrid, row: int, col: int, color: int):
        grid.values[row][col] = color
        grid.values[row][cols - 1 - col] = color
        grid.values[rows - 1 - row][col] = color
        grid.values[rows - 1 - row][cols - 1 - col] = color
    
    # Analyze the grid
    border_pattern = get_border_pattern(output_grid)
    central_area = get_central_area(output_grid)
    
    # Identify green cells and essential colors
    green_cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == 3]
    essential_colors = set(border_pattern + [color for row in central_area for color in row])
    essential_colors.discard(3)  # Remove green from essential colors
    
    # Remove green cells and extend patterns
    for r, c in green_cells:
        if not is_border(r, c):
            neighbors = get_neighborhood(output_grid, r, c)
            new_color = get_replacement_color(neighbors, essential_colors)
            apply_symmetrical(output_grid, r, c, new_color)
    
    # Preserve border pattern
    for i, color in enumerate(border_pattern):
        if i < cols:
            output_grid.values[0][i] = color
            output_grid.values[-1][i] = color
        elif i < cols + rows - 1:
            output_grid.values[i - cols + 1][0] = color
            output_grid.values[i - cols + 1][-1] = color
        else:
            output_grid.values[i - cols - rows + 2][0] = color
            output_grid.values[i - cols - rows + 2][-1] = color
    
    # Final consistency pass
    for r in range(rows):
        for c in range(cols):
            if not is_border(r, c) and output_grid.values[r][c] == 3:
                neighbors = get_neighborhood(output_grid, r, c)
                new_color = get_replacement_color(neighbors, essential_colors)
                apply_symmetrical(output_grid, r, c, new_color)
    
    return output_grid
