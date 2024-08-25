from rob_agi.colored_grid import ColoredGrid

def solve_67385a82(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing connected groups of 3s (green) to 8s (sky blue).
    Only horizontal and vertical connections are considered, not diagonal.
    Isolated 3s or 3s connected only diagonally remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with connected 3s changed to 8s.
    """
    def is_valid_cell(row, col, height, width):
        return 0 <= row < height and 0 <= col < width

    def find_connected_threes(row, col, height, width):
        connected = [(row, col)]
        stack = [(row, col)]
        visited = set([(row, col)])
        
        while stack:
            r, c = stack.pop()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if is_valid_cell(nr, nc, height, width) and (nr, nc) not in visited:
                    if input_grid.get_cell(nr, nc) == 3:
                        connected.append((nr, nc))
                        stack.append((nr, nc))
                        visited.add((nr, nc))
        
        return connected

    output = input_grid.deep_copy()
    height, width = input_grid.get_dimensions()

    for row in range(height):
        for col in range(width):
            if input_grid.get_cell(row, col) == 3:
                connected_region = find_connected_threes(row, col, height, width)
                if len(connected_region) > 1:
                    for r, c in connected_region:
                        output.set_cell(r, c, 8)

    return output
