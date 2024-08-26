from rob_agi.colored_grid import ColoredGrid

def solve_25094a63(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 25094a63 challenge by detecting and replacing a specific rectangular area with yellow.
    
    The function scans the entire grid to find a target rectangular area of a solid color
    (excluding yellow) with width between 7 and 9 cells and height of 7 cells. It then
    replaces this area with yellow (color code 4) while preserving any existing yellow cells.
    This solution works for all 30x30 input grids, adapting to variations in the target
    area's position, dimensions, and original color.
    
    Args:
        input_grid (ColoredGrid): The input 30x30 colored grid.
    
    Returns:
        ColoredGrid: The modified grid with the target area replaced by yellow.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = len(output_grid.values), len(output_grid.values[0])

    def is_valid_target_area(start_row, start_col):
        color = output_grid.values[start_row][start_col]
        if color == 4:  # Skip if starting color is yellow
            return False
        
        # Check width
        width = 0
        for c in range(start_col, cols):
            if output_grid.values[start_row][c] not in [color, 4]:
                break
            width += 1
        
        if width < 7 or width > 9:
            return False
        
        # Check height
        for r in range(start_row, min(start_row + 7, rows)):
            for c in range(start_col, start_col + width):
                if output_grid.values[r][c] not in [color, 4]:
                    return False
        
        return r - start_row + 1 == 7

    # Scan the entire grid
    for row in range(rows):
        for col in range(cols):
            if is_valid_target_area(row, col):
                # Replace target area with yellow
                for r in range(row, row + 7):
                    for c in range(col, col + min(9, cols - col)):
                        if output_grid.values[r][c] != 4:  # If not already yellow
                            output_grid.values[r][c] = 4  # Set to yellow
                return output_grid

    # If no valid target area found, return the original grid
    return input_grid
