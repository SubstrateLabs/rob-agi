from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_9f27f097.main import solve_9f27f097, determine_transformation, get_region_bounds

def create_test_grid():
    return ColoredGrid(values=[
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ])

def main():
    input_grid = create_test_grid()
    try:
        output_grid = solve_9f27f097(input_grid)
        print("solve_9f27f097 executed successfully")
    except Exception as e:
        print(f"Error in solve_9f27f097: {str(e)}")
        return

    # Simulate the call to determine_transformation
    source_region = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4),
                     (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4)]
    target_region = [(7, 4), (7, 5), (7, 6), (7, 7), (8, 4), (8, 5), (8, 6), (8, 7),
                     (9, 4), (9, 5), (9, 6), (9, 7), (10, 4), (10, 5), (10, 6), (10, 7)]

    source_bounds = get_region_bounds(source_region)
    target_bounds = get_region_bounds(target_region)

    try:
        transformation = determine_transformation(source_bounds, target_bounds)
        print(f"determine_transformation result: {transformation}")
    except Exception as e:
        print(f"Error in determine_transformation: {str(e)}")

if __name__ == "__main__":
    main()
