from rob_agi.colored_grid import ColoredGrid

def solve_b20f7c8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying specific rules to 5x5 regions.
    
    The transformation rules are:
    1. For left regions:
       - If solid, new_color = (original_color + 1) % 10
       - If patterned, fill with the color of the top-left pixel
    2. For right regions:
       - If solid, new_color = (original_color + 1) % 10
       - If patterned, fill with gray (5)
    
    The function identifies 5x5 regions in the middle and right side of the grid,
    analyzes them, and applies the appropriate transformation.
    """
    output_grid = input_grid.deep_copy()
    
    transformation_regions = [
        (1, 6), (9, 6),  # Left regions
        (1, 16), (9, 16)  # Right regions
    ]
    
    for row, col in transformation_regions:
        block = output_grid.extract_subgrid(row, col, 5, 5)
        is_left = col == 6
        transformed_block = transform_block(block, is_left)
        replace_5x5_block(output_grid, row, col, transformed_block)
    
    return output_grid

def is_solid_block(block: ColoredGrid) -> bool:
    return len(set(cell for row in block.values for cell in row)) == 1

def transform_block(block: ColoredGrid, is_left: bool) -> ColoredGrid:
    if is_solid_block(block):
        original_color = block.values[0][0]
        new_color = (original_color + 1) % 10
    else:
        new_color = block.values[0][0] if is_left else 5
    
    return ColoredGrid(values=[[new_color for _ in range(5)] for _ in range(5)])

def replace_5x5_block(grid: ColoredGrid, row: int, col: int, new_block: ColoredGrid):
    for i in range(5):
        for j in range(5):
            grid.values[row + i][col + j] = new_block.values[i][j]
