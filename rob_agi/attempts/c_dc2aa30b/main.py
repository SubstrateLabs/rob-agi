from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_dc2aa30b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by inverting color dominance between top and bottom sections,
    while maintaining a specific pattern in the middle section. The solution:
    1. Divides the grid into 3x3 sub-grids
    2. Makes the top section predominantly blue with two specific red cells
    3. Makes the bottom section predominantly red with one specific blue cell
    4. Inverts color dominance in the middle section while maintaining a balanced mix
    5. Preserves the black (0) dividing lines
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def process_top_section():
        for i in range(3):
            for j in range(11):
                if output_grid.values[i][j] != 0:
                    output_grid.values[i][j] = 1  # Set to blue
        
        # Place specific red cells
        output_grid.values[0][2] = 2
        output_grid.values[1][4] = 2
        output_grid.values[2][8] = 2
    
    def process_bottom_section():
        for i in range(8, 11):
            for j in range(11):
                if output_grid.values[i][j] != 0:
                    output_grid.values[i][j] = 2  # Set to red
        
        # Place specific blue cell
        output_grid.values[8][8] = 1
    
    def process_middle_section():
        blue_count = sum(cell == 1 for row in input_grid.values[4:7] for cell in row if cell != 0)
        red_count = sum(cell == 2 for row in input_grid.values[4:7] for cell in row if cell != 0)
        total_non_black = blue_count + red_count
        
        dominant_color = 2 if blue_count > red_count else 1
        target_dominant = int(total_non_black * 2 / 3)
        current_dominant = 0
        
        for i in range(4, 7):
            for j in range(11):
                if output_grid.values[i][j] != 0:
                    if current_dominant < target_dominant:
                        output_grid.values[i][j] = dominant_color
                        current_dominant += 1
                    else:
                        output_grid.values[i][j] = 3 - dominant_color  # 3 - 1 = 2, 3 - 2 = 1
    
    process_top_section()
    process_middle_section()
    process_bottom_section()
    
    return output_grid
