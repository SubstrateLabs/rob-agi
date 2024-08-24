from rob_agi.colored_grid import ColoredGrid

def solve_1cf80156(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the non-zero elements from the input grid while maintaining their relative positions.
    Removes surrounding zero (empty) rows and columns.
    
    1. Find the bounding box of non-zero elements.
    2. Extract the subgrid within the bounding box.
    3. Remove any remaining zero rows or columns.
    """
    def find_bounding_box(grid):
        rows, cols = len(grid), len(grid[0])
        min_row, min_col = rows, cols
        max_row, max_col = -1, -1
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0:
                    min_row = min(min_row, r)
                    min_col = min(min_col, c)
                    max_row = max(max_row, r)
                    max_col = max(max_col, c)
        
        return (min_row, min_col, max_row, max_col) if max_row >= min_row else None

    values = input_grid.values
    bbox = find_bounding_box(values)
    
    if not bbox:
        return ColoredGrid(values=[[]])
    
    min_row, min_col, max_row, max_col = bbox
    
    # Extract the subgrid within the bounding box
    subgrid = [row[min_col:max_col+1] for row in values[min_row:max_row+1]]
    
    # Remove any remaining zero rows or columns
    def remove_zero_rows(grid):
        return [row for row in grid if any(cell != 0 for cell in row)]
    
    def remove_zero_columns(grid):
        return list(map(list, zip(*remove_zero_rows(list(zip(*grid))))))
    
    result = remove_zero_columns(remove_zero_rows(subgrid))
    
    return ColoredGrid(values=result)
