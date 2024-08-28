from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_d492a647.main import solve_d492a647

def run_experiment():
    # Example input grid from test case
    input_grid = ColoredGrid(values=[
        [0, 0, 5, 0, 5, 5, 5, 0, 5, 0, 5, 5, 5],
        [5, 5, 0, 5, 0, 0, 5, 5, 0, 5, 5, 5, 5],
        [5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, 5, 0, 0, 1, 0, 0, 0, 0, 0, 5, 5],
        [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
        [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5],
        [0, 0, 5, 5, 0, 5, 0, 5, 0, 5, 5, 5, 5],
        [5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 0, 5, 0, 5, 5, 5, 5, 0, 5, 0, 5]
    ])

    output_grid = solve_d492a647(input_grid)

    print("Cells that changed color:")
    rows, cols = input_grid.get_dimensions()
    for row in range(rows):
        for col in range(cols):
            input_color = input_grid.values[row][col]
            output_color = output_grid.values[row][col]
            if input_color != output_color:
                print(f"Row {row}, Col {col}: {input_color} -> {output_color}")
                print(f"  Sum of indices: {row + col}")
                print(f"  Should change: {input_color == 0 and (row + col) % 2 == 1}")

    print("\nCells that shouldn't have changed but did:")
    for row in range(rows):
        for col in range(cols):
            input_color = input_grid.values[row][col]
            output_color = output_grid.values[row][col]
            if input_color != output_color and (input_color != 0 or (row + col) % 2 != 1):
                print(f"Row {row}, Col {col}: {input_color} -> {output_color}")

if __name__ == "__main__":
    run_experiment()
