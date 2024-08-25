from rob_agi.colored_grid import ColoredGrid

def solve_b7f8a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colors from the centers of supercells.
    The expansion rules depend on the color of the center and the border of the supercell:
    - In red (2) supercells:
      * Yellow (4) expands horizontally within the cell.
      * Blue (1) expands vertically beyond the cell, stopping at supercell borders.
      * Green (3) expands vertically only within the cell.
    - In green (3) supercells:
      * Yellow (4) and Sky (8) expand horizontally within the cell.
      * Blue (1) expands vertically within the supercell.
      * Red (2) does not expand.
    - In blue (1) supercells:
      * Green (3) expands horizontally within the cell and vertically beyond it, stopping at other colored cells or grid edges.
      * Red (2) does not expand.
    The grid structure (borders and frames) is preserved.
    """
    height, width = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Identify supercell size
    supercell_size = next(i for i in range(1, width) if input_grid.values[1][i] != 0)
    
    def get_supercell_border(row, col):
        return input_grid.values[row - row % supercell_size][col - col % supercell_size]
    
    def expand_horizontal(row, col, color, limit):
        for c in range(col - 1, max(col - limit, col - col % supercell_size), -1):
            if output_grid.values[row][c] in {0, 2}:
                output_grid.values[row][c] = color
            else:
                break
        for c in range(col + 1, min(col + limit, col - col % supercell_size + supercell_size)):
            if output_grid.values[row][c] in {0, 2}:
                output_grid.values[row][c] = color
            else:
                break

    def expand_vertical(row, col, color, up_limit, down_limit):
        for r in range(row - 1, max(row - up_limit, -1), -1):
            if output_grid.values[r][col] in {0, 2}:
                output_grid.values[r][col] = color
            else:
                break
        for r in range(row + 1, min(row + down_limit, height)):
            if output_grid.values[r][col] in {0, 2}:
                output_grid.values[r][col] = color
            else:
                break

    def expand_green_in_blue(row, col):
        expand_horizontal(row, col, 3, supercell_size // 2)
        expand_vertical(row, col, 3, height, height)
    
    # Process expansions
    for row in range(supercell_size // 2, height, supercell_size):
        for col in range(supercell_size // 2, width, supercell_size):
            color = input_grid.values[row][col]
            border = get_supercell_border(row, col)
            
            if border == 2:  # Red supercell
                if color == 4:  # Yellow
                    expand_horizontal(row, col, color, supercell_size // 2)
                elif color == 1:  # Blue
                    expand_vertical(row, col, color, height, height)
                elif color == 3:  # Green
                    expand_vertical(row, col, color, 1, 1)
            elif border == 3:  # Green supercell
                if color in {4, 8}:  # Yellow or Sky
                    expand_horizontal(row, col, color, supercell_size // 2)
                elif color == 1:  # Blue
                    expand_vertical(row, col, color, supercell_size // 2, supercell_size // 2)
            elif border == 1:  # Blue supercell
                if color == 3:  # Green
                    expand_green_in_blue(row, col)
    
    # Preserve grid structure
    for row in range(height):
        for col in range(width):
            if row % supercell_size == 0 or col % supercell_size == 0:
                output_grid.values[row][col] = input_grid.values[row][col]
    
    return output_grid
