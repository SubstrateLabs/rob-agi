from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e4075551(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a rectangular frame pattern based on the colors present.
    
    1. Identifies unique non-black (0) and non-red (2) colors in the input grid
    2. Determines frame dimensions based on the number of unique colors
    3. Calculates frame position to center it horizontally
    4. Assigns colors to different parts of the frame (top, left, right, bottom)
    5. Creates a new grid and draws the frame with assigned colors
    6. Fills the frame interior with gray (5)
    7. Draws a red (2) horizontal line in the middle of the frame
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    unique_colors = find_unique_colors(input_grid)
    frame_width = len(unique_colors) + 1
    frame_height = frame_width + 1
    start_row = 1
    start_col = (cols - frame_width) // 2
    color_assignments = assign_colors(unique_colors)
    
    # Draw frame borders
    draw_frame(output_grid, start_row, start_col, frame_height, frame_width, color_assignments)
    
    # Fill frame interior
    fill_frame(output_grid, start_row, start_col, frame_height, frame_width)
    
    # Draw red center line
    center_row = start_row + frame_height // 2
    draw_red_line(output_grid, center_row, start_col + 1, start_col + frame_width - 2)
    
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
    colors = unique_colors * (4 // len(unique_colors) + 1)  # Repeat colors if needed
    return {
        'top': colors[0],
        'left': colors[1],
        'right': colors[-2],
        'bottom': colors[-1]
    }

def draw_frame(grid: ColoredGrid, start_row: int, start_col: int, frame_height: int, frame_width: int, colors: dict):
    # Draw top and bottom borders
    draw_line(grid, start_row, start_col, start_col + frame_width - 1, colors['top'])
    draw_line(grid, start_row + frame_height - 1, start_col, start_col + frame_width - 1, colors['bottom'])
    
    # Draw left and right borders
    draw_line(grid, start_row + 1, start_col, start_row + frame_height - 2, colors['left'])
    draw_line(grid, start_row + 1, start_col + frame_width - 1, start_row + frame_height - 2, colors['right'])

def draw_horizontal_line(grid: ColoredGrid, row: int, start_col: int, end_col: int, color: int):
    for c in range(start_col, end_col + 1):
        grid.set_cell(row, c, color)

def draw_vertical_line(grid: ColoredGrid, col: int, start_row: int, end_row: int, color: int):
    for r in range(start_row, end_row + 1):
        grid.set_cell(r, col, color)

def draw_frame(grid: ColoredGrid, start_row: int, start_col: int, frame_height: int, frame_width: int, colors: dict):
    # Draw top and bottom borders
    draw_horizontal_line(grid, start_row, start_col, start_col + frame_width - 1, colors['top'])
    draw_horizontal_line(grid, start_row + frame_height - 1, start_col, start_col + frame_width - 1, colors['bottom'])
    
    # Draw left and right borders
    draw_vertical_line(grid, start_col, start_row + 1, start_row + frame_height - 2, colors['left'])
    draw_vertical_line(grid, start_col + frame_width - 1, start_row + 1, start_row + frame_height - 2, colors['right'])

def fill_frame(grid: ColoredGrid, start_row: int, start_col: int, frame_height: int, frame_width: int):
    for r in range(start_row + 1, start_row + frame_height - 1):
        for c in range(start_col + 1, start_col + frame_width - 1):
            grid.set_cell(r, c, 5)  # Gray

def draw_red_line(grid: ColoredGrid, row: int, start_col: int, end_col: int):
    draw_horizontal_line(grid, row, start_col, end_col, 2)  # Red
