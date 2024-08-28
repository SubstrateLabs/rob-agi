from rob_agi.colored_grid import ColoredGrid
from main import solve_a1b2c3d4

def test_plus_shape():
    input_grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    grid = ColoredGrid(values=input_grid)
    result = solve_a1b2c3d4(grid)
    
    print("Input grid:")
    for row in input_grid:
        print(row)
    
    print("\nOutput grid:")
    for row in result.values:
        print(row)

test_plus_shape()
