from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_903d1b4a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Simplify the pattern in the input grid by removing infrequent or isolated colors
    while preserving the main structure and symmetry. The solution involves:
    1. Analyzing the grid to identify main colors, structures, and symmetry.
    2. Processing the central area to simplify its pattern.
    3. Extending simplification outwards while maintaining symmetry.
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
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    neighbors.append(grid.values[r][c])
        return neighbors
    
    def get_most_frequent_color(colors: List[int], essential_colors: Set[int]) -> int:
        return Counter([c for c in colors if c in essential_colors]).most_common(1)[0][0]
    
    def is_border(row: int, col: int) -> bool:
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1
    
    def apply_symmetrical(grid: ColoredGrid, row: int, col: int, color: int):
        grid.values[row][col] = color
        grid.values[row][cols - 1 - col] = color
        grid.values[rows - 1 - row][col] = color
        grid.values[rows - 1 - row][cols - 1 - col] = color
    
    # Analyze the grid
    color_freq = get_color_frequency(output_grid)
    border_pattern = get_border_pattern(output_grid)
    central_area = get_central_area(output_grid)
    
    # Identify essential colors
    total_squares = rows * cols
    essential_colors = set(color for color, freq in color_freq.items() if freq >= total_squares * 0.05)
    essential_colors.update(set(border_pattern))
    
    # Process central area
    for r in range(len(central_area)):
        for c in range(len(central_area[0])):
            if central_area[r][c] not in essential_colors:
                neighbors = get_neighborhood(ColoredGrid(values=central_area), r, c)
                new_color = get_most_frequent_color(neighbors, essential_colors)
                central_area[r][c] = new_color
    
    # Apply central area changes to the output grid
    center_start_row = (rows - len(central_area)) // 2
    center_start_col = (cols - len(central_area[0])) // 2
    for r in range(len(central_area)):
        for c in range(len(central_area[0])):
            apply_symmetrical(output_grid, center_start_row + r, center_start_col + c, central_area[r][c])
    
    # Extend simplification outwards
    for r in range(rows):
        for c in range(cols):
            if not is_border(r, c) and output_grid.values[r][c] not in essential_colors:
                neighbors = get_neighborhood(output_grid, r, c)
                new_color = get_most_frequent_color(neighbors, essential_colors)
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
            if not is_border(r, c):
                neighbors = get_neighborhood(output_grid, r, c)
                if output_grid.values[r][c] not in neighbors:
                    new_color = get_most_frequent_color(neighbors, essential_colors)
                    apply_symmetrical(output_grid, r, c, new_color)
    
    return output_grid
