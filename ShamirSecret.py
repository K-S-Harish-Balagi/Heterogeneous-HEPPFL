import random
PRIME_Q = 6458713 # Prime number 

def calculate_Y(x, poly):
    y, temp = 0, 1
    for coeff in poly: 
        y = (y + (coeff * temp)) % PRIME_Q
        temp = (temp * x) % PRIME_Q
    return y

def generate_share(shamir_secret, points, threshold):
    poly = [shamir_secret]
    for _ in range(threshold - 1):
        poly.append(random.randint(1, PRIME_Q - 1))

    return {x: calculate_Y(x, poly) for x in points}

def reconstruct_secret(points):
    # Reconstructs the secret using Lagrange interpolation
    secret = 0
    
    for xi, yi in points.items():
        num = 1
        den = 1
        for xj in points.keys():
            if xi != xj:
                num *= -xj
                den *= (xi - xj)
        
        den_inv = pow(den, -1, PRIME_Q)  # finds the modulo inverse 
        secret = (secret + yi * num * den_inv) % PRIME_Q  

    return secret 
