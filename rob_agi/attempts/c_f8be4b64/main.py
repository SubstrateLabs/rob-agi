from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f8be4b64(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored centers into territories.
    
    1. Identifies colored centers (non-green, non-black cells adjacent to green cells).
    2. Sorts colored centers by color value in descending order.
    3. Creates vertical lines for each colored center.
    4. Creates horizontal lines for each colored center.
    5. Fills territories between lines.
    6. Preserves original green cells.
    7. Removes isolated green cells.
    8. Ensures all non-edge cells have a non-black color.
    9. Expands territories to fill the entire grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    colored_centers = find_colored_centers(input_grid)
    
    if not colored_centers:
        default_color = max(max(row) for row in input_grid.values)
        colored_centers = [(0, 0, default_color)]
    
    # Sort colored centers by color value in descending order
    colored_centers.sort(key=lambda x: x[2], reverse=True)
    
    # Create vertical lines
    create_vertical_lines(new_grid, colored_centers)
    
    # Create horizontal lines
    create_horizontal_lines(new_grid, colored_centers)
    
    # Fill territories
    fill_territories(new_grid)
    
    # Preserve original green cells
    preserve_green_cells(new_grid, input_grid)
    
    # Remove isolated green cells
    remove_isolated_green_cells(new_grid)
    
    # Ensure all non-edge cells have a non-black color
    fill_remaining_black_cells(new_grid)
    
    # Expand territories to fill the entire grid
    expand_territories(new_grid)
    
    return new_grid

def find_colored_centers(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    centers = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] not in [0, 3] and is_adjacent_to_green(grid, r, c):
                centers.append((r, c, grid.values[r][c]))
    return centers

def is_adjacent_to_green(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 3:
            return True
    return False

def create_vertical_lines(grid: ColoredGrid, centers: List[Tuple[int, int, int]]):
    rows, cols = grid.get_dimensions()
    for _, c, color in centers:
        for r in range(rows):
            if grid.values[r][c] == 0 or color > grid.values[r][c]:
                grid.values[r][c] = color

def create_horizontal_lines(grid: ColoredGrid, centers: List[Tuple[int, int, int]]):
    rows, cols = grid.get_dimensions()
    for r, c, color in centers:
        # Expand left
        for col in range(c - 1, -1, -1):
            if grid.values[r][col] == 0 or grid.values[r][col] == color:
                grid.values[r][col] = color
            else:
                break
        # Expand right
        for col in range(c + 1, cols):
            if grid.values[r][col] == 0 or grid.values[r][col] == color:
                grid.values[r][col] = color
            else:
                break

def fill_territories(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        current_color = 0
        for c in range(cols):
            if grid.values[r][c] != 0:
                current_color = grid.values[r][c]
            elif current_color != 0:
                grid.values[r][c] = current_color

def preserve_green_cells(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 3:
                new_grid.values[r][c] = 3

def remove_isolated_green_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 3:
                if not any(0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] == 3
                           for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                    adjacent_colors = [grid.values[r + dr][c + dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                       if 0 <= r + dr < rows and 0 <= c + dc < cols and grid.values[r + dr][c + dc] != 0]
                    if adjacent_colors:
                        grid.values[r][c] = max(adjacent_colors)

def fill_remaining_black_cells(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid.values[r][c] == 0:
                adjacent_colors = [grid.values[r + dr][c + dc] for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                   if grid.values[r + dr][c + dc] != 0]
                if adjacent_colors:
                    grid.values[r][c] = max(adjacent_colors)

def expand_territories(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        left_color = 0
        right_color = 0
        for c in range(cols):
            if grid.values[r][c] != 0:
                left_color = grid.values[r][c]
                break
        for c in range(cols - 1, -1, -1):
            if grid.values[r][c] != 0:
                right_color = grid.values[r][c]
                break
        for c in range(cols):
            if grid.values[r][c] == 0:
                if c < cols // 2:
                    grid.values[r][c] = left_color
                else:
                    grid.values[r][c] = right_color
