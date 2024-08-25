from rob_agi.colored_grid import ColoredGrid

def solve_73182012(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a 4x4 subgrid from the input grid, starting from the top-left corner of the non-zero pattern.
    
    The function finds the bounding box of the non-zero elements in the input grid,
    then extracts a 4x4 subgrid from the top-left corner of this bounding box.
    If the bounding box is smaller than 4x4, the remaining cells are filled with zeros.
    """
    def find_bounding_box(grid):
        rows, cols = grid.get_dimensions()
        min_row, max_row = rows, 0
        min_col, max_col = cols, 0
        
        for i in range(rows):
            for j in range(cols):
                if grid.get_cell(i, j) != 0:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)
        
        return min_row, min_col, min(min_row + 3, max_row), min(min_col + 3, max_col)

    top, left, bottom, right = find_bounding_box(input_grid)
    
    output_values = []
    for i in range(4):
        row = []
        for j in range(4):
            if top + i <= bottom and left + j <= right:
                row.append(input_grid.get_cell(top + i, left + j))
            else:
                row.append(0)
        output_values.append(row)
    
    return ColoredGrid(values=output_values)
