from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting and compacting non-background elements.

    1. Identifies the background color (most common color on the border)
    2. Extracts all non-background elements, including black (0)
    3. Creates a new compact grid containing only non-background elements
    4. Preserves relative positions, relationships, and shapes between elements
    5. Removes rows and columns that are entirely background color
    6. Returns the new compact grid

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: A new compact grid containing only non-background elements
    """
    def find_background_color(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        border = (
            grid.values[0] + grid.values[-1] +
            [row[0] for row in grid.values[1:-1]] +
            [row[-1] for row in grid.values[1:-1]]
        )
        return max(set(border), key=border.count)

    def extract_non_background_elements(grid: ColoredGrid, bg_color: int) -> List[Tuple[int, int, int]]:
        rows, cols = grid.get_dimensions()
        elements = []
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != bg_color:
                    elements.append((grid.values[r][c], r, c))
        return elements

    def create_compact_grid(elements: List[Tuple[int, int, int]], bg_color: int) -> List[List[int]]:
        if not elements:
            return [[]]
        
        min_row = min(elem[1] for elem in elements)
        min_col = min(elem[2] for elem in elements)
        max_row = max(elem[1] for elem in elements)
        max_col = max(elem[2] for elem in elements)
        
        # Initialize grid with background color
        grid = [[bg_color for _ in range(max_col - min_col + 1)] for _ in range(max_row - min_row + 1)]
        
        for color, r, c in elements:
            grid[r - min_row][c - min_col] = color
        
        return grid

    def remove_background_rows_and_columns(grid: List[List[int]], bg_color: int) -> List[List[int]]:
        # Remove background rows
        grid = [row for row in grid if any(cell != bg_color for cell in row)]
        
        # Remove background columns
        cols_to_keep = [col for col in range(len(grid[0])) if any(row[col] != bg_color for row in grid)]
        grid = [[row[col] for col in cols_to_keep] for row in grid]
        
        return grid

    bg_color = find_background_color(input_grid)
    non_bg_elements = extract_non_background_elements(input_grid, bg_color)
    compact_grid = create_compact_grid(non_bg_elements, bg_color)
    final_grid = remove_background_rows_and_columns(compact_grid, bg_color)

    return ColoredGrid(values=final_grid)
