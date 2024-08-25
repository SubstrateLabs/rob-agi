from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_7c9b52a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting non-background elements within their bounding box.

    1. Identifies the background color
    2. Finds the bounding box of non-background elements
    3. Extracts the contents of the bounding box
    4. Removes empty rows and columns
    5. Returns the new compact grid containing only the extracted region

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: A new grid containing the extracted non-background region
    """
    def find_background_color(grid: ColoredGrid) -> int:
        rows, cols = grid.get_dimensions()
        border = (
            grid.values[0] + grid.values[-1] +
            [row[0] for row in grid.values[1:-1]] +
            [row[-1] for row in grid.values[1:-1]]
        )
        return max(set(border), key=border.count)

    def find_bounding_box(grid: ColoredGrid, bg_color: int) -> Tuple[int, int, int, int]:
        rows, cols = grid.get_dimensions()
        min_row, max_row = rows, 0
        min_col, max_col = cols, 0

        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != bg_color:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

        return min_row, min_col, max_row, max_col

    def extract_bounding_box(grid: ColoredGrid, bbox: Tuple[int, int, int, int]) -> List[List[int]]:
        min_row, min_col, max_row, max_col = bbox
        return [row[min_col:max_col+1] for row in grid.values[min_row:max_row+1]]

    def remove_empty_rows_cols(grid: List[List[int]], bg_color: int) -> List[List[int]]:
        # Remove empty rows
        grid = [row for row in grid if any(cell != bg_color for cell in row)]
        
        # Remove empty columns
        if not grid:
            return grid
        
        cols_to_keep = [
            col for col in range(len(grid[0]))
            if any(row[col] != bg_color for row in grid)
        ]
        
        return [[row[col] for col in cols_to_keep] for row in grid]

    bg_color = find_background_color(input_grid)
    bbox = find_bounding_box(input_grid, bg_color)
    extracted_grid = extract_bounding_box(input_grid, bbox)
    final_grid = remove_empty_rows_cols(extracted_grid, bg_color)

    return ColoredGrid(values=final_grid)
