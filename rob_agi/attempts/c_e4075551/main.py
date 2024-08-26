from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e4075551(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a rectangular frame pattern based on the colors present.
    
    1. Identifies unique non-black and non-red colors in the input grid
    2. Determines frame dimensions based on the number of unique colors
    3. Calculates frame position to center it horizontally
    4. Assigns colors to different parts of the frame
    5. Draws the frame with assigned colors
    6. Fills the frame interior with gray
    7. Draws a red center line
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    unique_colors = find_unique_colors(input_grid)
    frame_width = len(unique_colors) + 1
    frame_height = frame_width + 2
    start_row = 1
    start_col = (cols - frame_width) // 2
    color_assignments = assign_colors(unique_colors)
    
    # Draw horizontal lines
    draw_line(output_grid, start_row, start_col, start_col + frame_width - 1, color_assignments['top'])
    draw_line(output_grid, start_row + frame_height - 1, start_col, start_col + frame_width - 1, color_assignments['bottom'])
    
    # Draw vertical lines
    draw_line(output_grid, start_row + 1, start_col, start_row + frame_height - 2, color_assignments['left'])
    draw_line(output_grid, start_row + 1, start_col + frame_width - 1, start_row + frame_height - 2, color_assignments['right'])
    
    # Fill frame interior
    fill_frame(output_grid, start_row, start_col, frame_height, frame_width)
    
    # Draw center line
    center_row = start_row + frame_height // 2
    draw_line(output_grid, center_row, start_col + 1, start_col + frame_width - 2, 2)  # Red center line
    
    return output_grid

def find_unique_colors(grid: ColoredGrid) -> List[int]:
    unique_colors = set()
    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            color = grid.get_cell(r, c)
            if color != 0 and color != 2:
                unique_colors.add(color)
    return sorted(list(unique_colors))

def assign_colors(unique_colors: List[int]) -> dict:
    if len(unique_colors) < 4:
        unique_colors = unique_colors * 4  # Repeat colors if less than 4
    return {
        'top': unique_colors[-1],
        'bottom': unique_colors[-2],
        'left': unique_colors[0],
        'right': unique_colors[1]
    }

def draw_line(grid: ColoredGrid, start_row: int, start_col: int, end: int, color: int):
    if start_row == end:  # Horizontal line
        for c in range(start_col, end + 1):
            grid.set_cell(start_row, c, color)
    else:  # Vertical line
        for r in range(start_row, end + 1):
            grid.set_cell(r, start_col, color)

def fill_frame(grid: ColoredGrid, start_row: int, start_col: int, frame_height: int, frame_width: int):
    for r in range(start_row + 1, start_row + frame_height - 1):
        for c in range(start_col + 1, start_col + frame_width - 1):
            grid.set_cell(r, c, 5)  # Gray
