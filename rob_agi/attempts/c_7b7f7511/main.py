from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_7b7f7511(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts the smallest non-repeating subgrid from the input grid.
    
    The function identifies the repeating pattern in both horizontal and vertical directions,
    extracts unique rows and columns, and combines them to form the smallest non-repeating subgrid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be processed.
    
    Returns:
    ColoredGrid: The smallest non-repeating subgrid extracted from the input.
    """
    def get_unique_elements(elements: List) -> List:
        seen = set()
        unique = []
        for element in elements:
            element_tuple = tuple(element)
            if element_tuple in seen:
                break
            seen.add(element_tuple)
            unique.append(element)
        return unique

    def transpose(grid: List[List[int]]) -> List[List[int]]:
        return list(map(list, zip(*grid)))

    # Extract unique rows
    unique_rows = get_unique_elements(input_grid.values)
    
    # Extract unique columns from the unique rows
    transposed_unique = transpose(unique_rows)
    unique_cols = get_unique_elements(transposed_unique)
    
    # Transpose back to get the final grid
    final_grid = transpose(unique_cols)

    return ColoredGrid(values=final_grid)
