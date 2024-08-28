from rob_agi.colored_grid import ColoredGrid

def solve_b7f8a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colors from the centers of supercells.
    The expansion rules depend on the color of the center and the border of the supercell:
    - Yellow (4) always expands horizontally within the supercell.
    - In red (2) supercells:
      * Blue (1) expands vertically beyond the cell, stopping at supercell borders.
      * Green (3) expands vertically only within the cell.
    - In green (3) supercells:
      * Sky (8) expands horizontally within the supercell and to adjacent empty spaces.
      * Blue (1) expands vertically within the supercell.
    - In blue (1) supercells:
      * Green (3) expands horizontally within the supercell and vertically beyond it, connecting with other green cells.
    The grid structure (borders and frames) is preserved.
    """
    height, width = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    # Identify supercell size
    supercell_size = next(i for i in range(1, width) if input_grid.values[1][i] != 0)
    
    def get_supercell_border(row, col):
        return input_grid.values[row - row % supercell_size][col - col % supercell_size]
    
    def expand_horizontal(row, col, color, within_supercell=True):
        left = col - col % supercell_size if within_supercell else max(0, col - 1)
        right = min(width, (col - col % supercell_size + supercell_size) if within_supercell else col + 2)
        for c in range(left, right):
            if c != col and output_grid.values[row][c] == 0:
                output_grid.values[row][c] = color

    def expand_vertical(row, col, color, within_supercell=False):
        top = row - row % supercell_size if within_supercell else 0
        bottom = min(height, (row - row % supercell_size + supercell_size) if within_supercell else height)
        for r in range(top, bottom):
            if r != row and output_grid.values[r][col] == 0:
                output_grid.values[r][col] = color
            elif not within_supercell and output_grid.values[r][col] != 0 and output_grid.values[r][col] != color:
                break

    def expand_green_in_blue(row, col):
        expand_horizontal(row, col, 3, within_supercell=True)
        expand_vertical(row, col, 3, within_supercell=False)
    
    # Process expansions
    for row in range(supercell_size // 2, height, supercell_size):
        for col in range(supercell_size // 2, width, supercell_size):
            color = input_grid.values[row][col]
            border = get_supercell_border(row, col)
            
            if color == 4:  # Yellow
                expand_horizontal(row, col, color)
            
            if border == 2:  # Red supercell
                if color == 1:  # Blue
                    expand_vertical(row, col, color)
                elif color == 3:  # Green
                    expand_vertical(row, col, color, within_supercell=True)
            elif border == 3:  # Green supercell
                if color == 8:  # Sky
                    expand_horizontal(row, col, color, within_supercell=False)
                elif color == 1:  # Blue
                    expand_vertical(row, col, color, within_supercell=True)
            elif border == 1:  # Blue supercell
                if color == 3:  # Green
                    expand_green_in_blue(row, col)
    
    # Connect vertical green expansions in blue supercells
    for col in range(supercell_size // 2, width, supercell_size):
        green_cells = []
        for row in range(height):
            if get_supercell_border(row, col) == 1 and output_grid.values[row][col] == 3:
                green_cells.append(row)
        for i in range(len(green_cells) - 1):
            start, end = green_cells[i], green_cells[i + 1]
            for r in range(start + 1, end):
                if output_grid.values[r][col] == 0:
                    output_grid.values[r][col] = 3
    
    # Connect horizontal yellow expansions
    for row in range(supercell_size // 2, height, supercell_size):
        yellow_cells = []
        for col in range(width):
            if output_grid.values[row][col] == 4:
                yellow_cells.append(col)
        for i in range(len(yellow_cells) - 1):
            start, end = yellow_cells[i], yellow_cells[i + 1]
            for c in range(start + 1, end):
                if output_grid.values[row][c] == 0:
                    output_grid.values[row][c] = 4
    
    # Preserve grid structure
    for row in range(height):
        for col in range(width):
            if row % supercell_size == 0 or col % supercell_size == 0:
                output_grid.values[row][col] = input_grid.values[row][col]
    
    return output_grid
