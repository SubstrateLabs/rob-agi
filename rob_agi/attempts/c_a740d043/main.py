from rob_agi.colored_grid import ColoredGrid

def solve_a740d043(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a740d043 challenge by extracting non-background elements.
    
    This function:
    1. Identifies the bounding box of non-background (non-1) elements.
    2. Extracts the region of interest defined by the bounding box.
    3. Transforms the extracted region by replacing background values (1) with 0.
    4. Removes any rows that consist entirely of zeros.
    5. Preserves all columns, including those that may consist entirely of zeros.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid containing only the relevant pattern.
    """
    def find_bounding_box(grid):
        rows, cols = len(grid), len(grid[0])
        top, bottom, left, right = rows, -1, cols, -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 1:
                    top = min(top, r)
                    bottom = max(bottom, r)
                    left = min(left, c)
                    right = max(right, c)
        return top, bottom, left, right

    def extract_and_transform(grid, bbox):
        top, bottom, left, right = bbox
        result = []
        for r in range(top, bottom + 1):
            row = [0 if grid[r][c] == 1 else grid[r][c] for c in range(left, right + 1)]
            if any(val != 0 for val in row):
                result.append(row)
        return result

    # Find the bounding box
    bbox = find_bounding_box(input_grid.values)
    
    # Extract, transform, and remove empty rows
    transformed_grid = extract_and_transform(input_grid.values, bbox)
    
    return ColoredGrid(values=transformed_grid)
