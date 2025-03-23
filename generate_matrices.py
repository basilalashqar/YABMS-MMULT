#!/usr/bin/env python3
import numpy as np
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python generate_matrices.py M N P")
        sys.exit(1)
    
    # Parse command-line arguments
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    P = int(sys.argv[3])
    
    # Generate random matrices A and B
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(N, P).astype(np.float32)
    
    # Compute the matrix multiplication result R = A * B
    R = np.dot(A, B)
    
    # Write matrices to binary files
    A.tofile("A.bin")
    B.tofile("B.bin")
    R.tofile("python_ref.bin")
    
    print(f"Generated A.bin with shape {A.shape}, B.bin with shape {B.shape}, and python_ref.bin with shape {R.shape}.")

if __name__ == "__main__":
    main()

