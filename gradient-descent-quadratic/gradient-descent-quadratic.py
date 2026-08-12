def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Run the gradient descent loop
    for _ in range(steps):
        # 1. Calculate the analytical gradient at the current x0: f'(x) = 2ax + b
        gradient = 2 * a * x0 + b
        
        # 2. Update x0 by moving in the opposite direction of the gradient
        x0 = x0 - lr * gradient
        
    return x0
