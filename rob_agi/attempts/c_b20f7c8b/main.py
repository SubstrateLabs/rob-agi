from rob_agi.colored_grid import ColoredGrid

def solve_b20f7c8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific rules to 5x5 regions.
    
    The transformation rules are:
    1. For left regions:
       - If solid, new_color = (original_color + 1) % 10
       - If patterned, fill with the first non-zero color encountered
    2. For right regions:
       - If solid, new_color = (original_color + 3) % 10
       - If patterned, fill with gray (5)
    
    The function applies the transformation twice:
    - First pass: Transform all four 5x5 regions
    - Second pass: Transform only the bottom two regions
    
    The transformation preserves the original pattern for non-solid blocks in left regions.
    """
    output_grid = input_grid.deep_copy()
    
    transformation_regions = [
        (1, 1), (1, 16),  # Top regions
        (9, 1), (9, 16)   # Bottom regions
    ]
    
    # First pass: Transform all regions
    for row, col in transformation_regions:
        transform_region(output_grid, row, col)
    
    # Second pass: Transform only bottom regions
    for row, col in transformation_regions[2:]:
        transform_region(output_grid, row, col)
    
    return output_grid

def is_solid_block(block: ColoredGrid) -> bool:
    non_zero_colors = set(cell for row in block.values for cell in row if cell != 0)
    return len(non_zero_colors) == 1

def get_first_non_zero_color(block: ColoredGrid) -> int:
    for row in block.values:
        for cell in row:
            if cell != 0:
                return cell
    return 0  # Default to 0 if all cells are zero

def transform_block(block: ColoredGrid, is_left: bool) -> ColoredGrid:
    if is_solid_block(block):
        original_color = get_first_non_zero_color(block)
        new_color = (original_color + 1) % 10 if is_left else (original_color + 3) % 10
        return ColoredGrid(values=[[new_color for _ in range(5)] for _ in range(5)])
    else:
        if is_left:
            return block  # Preserve the original pattern for non-solid left blocks
        else:
            return ColoredGrid(values=[[5 for _ in range(5)] for _ in range(5)])

def transform_region(grid: ColoredGrid, row: int, col: int):
    block = grid.extract_subgrid(row, col, 5, 5)
    is_left = col == 1
    transformed_block = transform_block(block, is_left)
    for i in range(5):
        for j in range(5):
            grid.values[row + i][col + j] = transformed_block.values[i][j]
