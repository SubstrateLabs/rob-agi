from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_3194b014(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 3194b014 challenge by finding the most significant rectangular area in the input grid
    and extracting a 3x3 grid from it.

    The solution follows these steps:
    1. Find all rectangular areas of each color in the input grid.
    2. Calculate a significance score for each rectangle based on its area and how close it is to being square.
    3. Identify the most significant rectangle.
    4. Extract a 3x3 grid from the most significant rectangle.
    5. Return the extracted 3x3 grid as a new ColoredGrid object.

    Args:
    input_grid (ColoredGrid): The input grid to be processed.

    Returns:
    ColoredGrid: A 3x3 grid extracted from the most significant rectangular area.
    """
    rectangles = find_all_rectangles(input_grid)
    most_significant = max(rectangles, key=calculate_significance_score)
    result_grid = extract_3x3_grid(input_grid, most_significant)
    return ColoredGrid(values=result_grid)

def find_all_rectangles(grid: ColoredGrid) -> List[Tuple[int, int, int, int, int]]:
    rectangles = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.get_cell(r, c)
            for height in range(1, rows - r + 1):
                for width in range(1, cols - c + 1):
                    if all(grid.get_cell(r+i, c+j) == color 
                           for i in range(height) for j in range(width)):
                        rectangles.append((color, r, c, height, width))
    return rectangles

def calculate_significance_score(rectangle: Tuple[int, int, int, int, int]) -> float:
    _, _, _, height, width = rectangle
    area = height * width
    square_factor = min(height, width) / max(height, width)
    return area * square_factor

def extract_3x3_grid(input_grid: ColoredGrid, rectangle: Tuple[int, int, int, int, int]) -> List[List[int]]:
    color, top, left, height, width = rectangle
    if height >= 3 and width >= 3:
        center_y = top + height // 2
        center_x = left + width // 2
        return [input_grid.values[r][center_x-1:center_x+2] for r in range(center_y-1, center_y+2)]
    else:
        result = [[color for _ in range(3)] for _ in range(3)]
        start_r = 1 - min(height, 3) // 2
        start_c = 1 - min(width, 3) // 2
        for r in range(max(height, 3)):
            for c in range(max(width, 3)):
                if 0 <= start_r + r < 3 and 0 <= start_c + c < 3:
                    result[start_r + r][start_c + c] = input_grid.get_cell(top + r, left + c)
        return result
