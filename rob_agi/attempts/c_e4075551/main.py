from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e4075551(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a rectangular frame pattern based on the colors present.
    
    1. Identifies unique non-black colors in the input grid
    2. Determines frame dimensions based on the number of unique colors
    3. Assigns colors to different parts of the frame
    4. Draws the frame with assigned colors
    5. Fills the frame interior with gray
    6. Draws a red center line
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    unique_colors = find_unique_colors(input_grid)
    frame_width = len(unique_colors) + 4
    color_assignments = assign_colors(unique_colors)
    
    # Draw horizontal lines
    draw_line(output_grid, 1, 2, 1, frame_width + 1, color_assignments['top'])
    draw_line(output_grid, 13, 2, 13, frame_width + 1, color_assignments['bottom'])
    
    # Draw vertical lines
    draw_line(output_grid, 2, 2, 12, 2, color_assignments['left'])
    draw_line(output_grid, 2, frame_width + 1, 12, frame_width + 1, color_assignments['right'])
    
    # Fill frame interior
    fill_frame(output_grid, frame_width)
    
    # Draw center line
    draw_line(output_grid, 7, 3, 7, frame_width, 2)  # Red center line
    
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
    return {
        'top': unique_colors[-1],
        'bottom': unique_colors[-2],
        'left': unique_colors[0],
        'right': unique_colors[1] if len(unique_colors) > 1 else unique_colors[0]
    }

def draw_line(grid: ColoredGrid, start_row: int, start_col: int, end_row: int, end_col: int, color: int):
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            grid.set_cell(r, c, color)

def fill_frame(grid: ColoredGrid, frame_width: int):
    for r in range(2, 13):
        for c in range(3, frame_width + 1):
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 5)  # Gray
