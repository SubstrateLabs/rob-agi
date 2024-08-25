from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_a406ac07(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding color regions from the right column and bottom row.
    
    1. Extracts color information from the right column and bottom row.
    2. Merges this information to create a list of (color, width, height) tuples.
    3. Places expanded rectangles in the output grid, starting from the top-left.
    4. Restores the original L-shape from the input grid.
    
    The function maintains the L-shape of colored cells while expanding each color
    into rectangles within the available space.
    """
    # Extract right column and bottom row
    right_col = [row[-1] for row in input_grid.values[:-1]]
    bottom_row = input_grid.values[-1][:-1]
    corner = input_grid.values[-1][-1]

    # Process right column
    col_data = process_sequence(right_col)
    
    # Process bottom row
    row_data = process_sequence(bottom_row)
    
    # Merge data
    merged_data = merge_data(col_data, row_data, corner)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(10)] for _ in range(10)])
    
    # Place expanded rectangles
    current_row, current_col = 0, 0
    for color, width, height in merged_data:
        available_width = min(width, 10 - current_col)
        available_height = min(height, 10 - current_row)
        for r in range(current_row, current_row + available_height):
            for c in range(current_col, current_col + available_width):
                output.values[r][c] = color
        current_row += available_height
        current_col += available_width
        if current_col >= 9 or current_row >= 9:
            break
    
    # Restore L-shape
    for i in range(10):
        output.values[i][-1] = input_grid.values[i][-1]
        output.values[-1][i] = input_grid.values[-1][i]
    
    return output

def process_sequence(seq: List[int]) -> List[Tuple[int, int]]:
    """Process a sequence of colors into (color, count) pairs."""
    result = []
    current_color = seq[0]
    count = 1
    for color in seq[1:]:
        if color == current_color:
            count += 1
        else:
            result.append((current_color, count))
            current_color = color
            count = 1
    result.append((current_color, count))
    return result

def merge_data(col_data: List[Tuple[int, int]], row_data: List[Tuple[int, int]], corner: int) -> List[Tuple[int, int, int]]:
    """Merge column and row data into (color, width, height) tuples."""
    color_dict = {}
    for color, height in col_data:
        color_dict[color] = [1, height]
    for color, width in row_data:
        if color in color_dict:
            color_dict[color][0] = width
        else:
            color_dict[color] = [width, 1]
    if corner not in color_dict:
        color_dict[corner] = [1, 1]
    return [(color, data[0], data[1]) for color, data in color_dict.items()]
