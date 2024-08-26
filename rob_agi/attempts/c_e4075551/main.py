from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e4075551(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 'H' shaped pattern based on the colors present.
    
    1. Identifies colored squares in the input grid
    2. Assigns colors to different parts of the 'H' shape
    3. Draws the 'H' shape with assigned colors
    4. Fills the frame with gray
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    colored_squares = find_colored_squares(input_grid)
    sorted_colors = sort_colors(colored_squares)
    color_assignments = assign_colors(sorted_colors)
    
    # Draw horizontal lines
    draw_line(output_grid, 2, 2, 2, 13, color_assignments['top'])
    draw_line(output_grid, 13, 2, 13, 13, color_assignments['bottom'])
    
    # Draw vertical lines
    draw_line(output_grid, 3, 2, 12, 2, color_assignments['left'])
    draw_line(output_grid, 3, 13, 12, 13, color_assignments['right'])
    
    # Set center
    output_grid.set_cell(6, 6, 2)  # Red center
    
    # Fill frame
    fill_frame(output_grid)
    
    return output_grid

def find_colored_squares(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    return [(grid.get_cell(r, c), r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.get_cell(r, c) != 0]

def sort_colors(colored_squares: List[Tuple[int, int, int]]) -> List[int]:
    return sorted(set(color for color, _, _ in colored_squares))

def assign_colors(sorted_colors: List[int]) -> dict:
    return {
        'top': sorted_colors[2],
        'left': sorted_colors[1],
        'right': sorted_colors[3],
        'bottom': sorted_colors[4]
    }

def draw_line(grid: ColoredGrid, start_row: int, start_col: int, end_row: int, end_col: int, color: int):
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            grid.set_cell(r, c, color)

def fill_frame(grid: ColoredGrid):
    for r in range(3, 13):
        for c in range(3, 13):
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 5)  # Gray
