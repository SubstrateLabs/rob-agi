from rob_agi.colored_grid import ColoredGrid

def solve_3906de3d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving red cells (2) upwards, stopping at blue cells (1) or the 4th row from the top.
    Red cells are stacked from the bottom up in each column, filling gaps between blue cells.
    The bottom three rows are always cleared of red cells.
    Blue cells act as barriers, and red cells cannot pass through them.
    
    1. Process each column independently.
    2. Find blue cells and gaps between them.
    3. Move red cells upwards, filling gaps from bottom to top, but not higher than the 4th row from the top.
    4. Clear any remaining red cells in the bottom three rows.
    5. Ensure red cells do not pass through blue cells.
    """
    grid = input_grid.deep_copy()
    height, width = grid.get_dimensions()
    
    def process_column(col):
        blue_cells = [row for row in range(height) if grid.get_cell(row, col) == 1]
        red_cells = [row for row in range(height) if grid.get_cell(row, col) == 2]
        
        # Clear original red cells
        for row in red_cells:
            grid.set_cell(row, col, 0)
        
        # Find gaps between blue cells
        gaps = []
        if blue_cells:
            gaps.append((0, blue_cells[0]))
            for i in range(len(blue_cells) - 1):
                gaps.append((blue_cells[i] + 1, blue_cells[i+1]))
            gaps.append((blue_cells[-1] + 1, height))
        else:
            gaps.append((0, height))
        
        # Stack red cells in gaps from bottom up
        red_count = len(red_cells)
        for start, end in reversed(gaps):
            for row in range(end - 1, max(start, 3) - 1, -1):  # Stop at 4th row from top or start of gap
                if red_count > 0 and row < height - 3:
                    grid.set_cell(row, col, 2)
                    red_count -= 1
                if red_count == 0:
                    break
            if red_count == 0:
                break
    
    # Process each column
    for col in range(width):
        process_column(col)
    
    return grid
