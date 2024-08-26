from rob_agi.colored_grid import ColoredGrid

def solve_1c0d0a4b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by outlining sky blue (8) regions with red (2).
    
    This function identifies all connected regions of sky blue color in the input grid,
    then creates a new grid where the outlines of these regions are marked in red.
    The outline includes all cells of a region that are either on the edge of the grid
    or adjacent to a black (0) cell.
    
    Args:
    input_grid (ColoredGrid): The input grid containing sky blue regions.
    
    Returns:
    ColoredGrid: A new grid with red outlines of the original sky blue regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    regions = input_grid.find_connected_regions(8)  # Find all sky blue regions
    
    for region in regions:
        outline = set()
        min_row = min(cell[0] for cell in region)
        max_row = max(cell[0] for cell in region)
        min_col = min(cell[1] for cell in region)
        max_col = max(cell[1] for cell in region)
        
        for r, c in region:
            # Check if cell is on the edge of the region
            if r == min_row or r == max_row or c == min_col or c == max_col:
                outline.add((r, c))
            else:
                # Check adjacent cells
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols) or input_grid.values[nr][nc] == 0:
                        outline.add((r, c))
                        break
        
        # Set outline cells to red in the output grid
        for r, c in outline:
            output_grid.values[r][c] = 2
    
    return output_grid
