import random

def estimate_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        # check if point is inside the circle
        if x**2 + y**2 <= 1:
            inside_circle += 1

    # pi = 4 x (points inside circle)
    return 4 * inside_circle / num_samples

print(estimate_pi(10000000))