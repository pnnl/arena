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

TOTAL = 8
SIZE  = TOTAL*TOTAL/4
global_A = [ [0 for _ in range( TOTAL ) ] for _ in range( TOTAL ) ]
global_result = [0 for _ in range( TOTAL ) ]
local_X = [0 for _ in range( TOTAL ) ]
for i in range( TOTAL ):
  for j in range( TOTAL ):
    if (i+j)%4 == 0:
      global_A[i][j] = i+j+1
    else:
      global_A[i][j] = 0
  
for i in range( TOTAL ):
  local_X[i] = i + 1
  global_result[i] = 0

print( "A: ", global_A )
print( "X: ", local_X )
print( "R: ", global_result)

out = np.dot( global_A, local_X )
print( out )
