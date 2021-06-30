# ========================================================================
# lu.py
# ========================================================================
# Serial implementation of SPMV.
#
# Author : Cheng Tan
#   Date : February 18, 2020
import numpy as np
import math

#N = np.random.randint(1, 10)
N = 8
A = 1.0 * np.random.randint(0, 10, size=(N,N))

# ------------------------------------------------------------------------
# Conventional LU 
# ------------------------------------------------------------------------
def lu():
  for i in range( N ):
    for j in range( i ):
      for k in range( j ):
        A[i][j] -= A[i][k] * A[k][j]
      A[i][j] /= A[j][j]*1.0
    for j in range( i, N):
      for k in range( i ):
        A[i][j] -= A[i][k] * A[k][j]

# ------------------------------------------------------------------------
# Transformed LU of single loop implementation
# ------------------------------------------------------------------------
def lu_transformed():
  for i in range( N ):
    for x in range( N * i ):
      j = x / i
      k = x % i
      if j < i:
        if k < j:
          A[i][j] -= A[i][k] * A[k][j]
        if k == i-1:
          A[i][j] /= A[j][j]*1.0
      else:
        if k < i:
          A[i][j] -= A[i][k] * A[k][j]

 # ------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------
def init():
  for i in range( N ):
    for j in range( i+1 ):
      A[i][j] = -(j % N) / (N*1.0) + 1;
    for j in range( i+1, N ):
      A[i][j] = 0
    A[i][i] = 1
  B = [ [0 for _ in range( N ) ] for _ in range( N ) ]
  for r in range( N ):
    for s in range( N ):
      B[r][s] = 0.0
  for t in range( N ):
    for r in range( N ):
      for s in range( N ):
        B[r][s] += A[r][t] * A[s][t]
  for r in range( N ):
    for s in range( N ):
      A[r][s] = B[r][s]

init()
print( "-" * 74 )
print( "init A for lu: " )
print( A )
print( "-" * 74 )
print( "conventional lu: " )
lu()
print( A )
print( "-" * 74 )
print( "transformed lu: " )
init()
lu_transformed()
print( A )

