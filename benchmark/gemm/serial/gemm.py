import numpy as np

#I     = np.random.randint(1, 10)
#K     = np.random.randint(1, 10)
#J     = np.random.randint(1, 10)
#A     = np.random.randint(0, 10, size=(I,K))
#B     = np.random.randint(0, 10, size=(K,J))
#C     = np.random.randint(0, 10, size=(I,J))
#alpha = np.random.randint(0, 10)
#beta  = np.random.randint(0, 10)
#OUT   = [ [0 for _ in range( J ) ] for _ in range( I ) ]

I = 16
J = I
K = I
A = np.random.randint(0, 10, size=(I,K))
B = np.random.randint(0, 10, size=(K,J))
C = np.random.randint(0, 10, size=(I,J))
tmp = 0
for i in range( I ):
  for j in range( J ):
    A[i][j] = tmp
    B[i][j] = j
    C[i][j] = i
    print("tmp: ", tmp, i, j)
    tmp += 1
  tmp = tmp - (I - 1)
alpha = 2
beta = 1
OUT = np.random.randint(0, 10, size=(I,J))
#OUT = [ [ 0 ] * I ] * I
#[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]

def gemm():
  for i in range( I ):
    for j in range( J ):
      for k in range( K ):
        OUT[i][j] += A[i][k] * B[k][j]
        if k == K - 1:
          OUT[i][j] *= alpha
          OUT[i][j] += C[i][j]*beta
  print( np.dot(1, OUT) )

def gemm_transformed():
  for x in range( I * J * K ):
    i = ( x / ( K * J ) ) % I
    j = ( x / K ) % J
    k = x % K
    OUT[i][j] += A[i][k] * B[k][j]
    if k == K-1:
      OUT[i][j] *= alpha
      OUT[i][j] += C[i][j]*beta
  print( np.dot(1, OUT) )

def init():
  for i in range( I ):
    for j in range( J ):
      OUT[i][j] = 0

init()
print( "-" * 74 )
print( "conventional gemm: " )
gemm()
init()
print( "-" * 74 )
print( "loop transformed gemm: " )
gemm_transformed()
print( "-" * 74 )
print( "numpy library gemm: " )
out = np.dot(alpha, np.dot(A, B)) + np.dot(beta, C)
print( out )
