# ========================================================================
# bfs.py
# ========================================================================
# Serial implementation of BFS with CSR format.
#
# Author : Cheng Tan
#   Date : April 4, 2020

num_vertice = 0
num_edge    = 0
root = 0

f = open("../data/graph.txt", "r")
num_line = 0
lines = f.readlines()
num_lines   = len( lines )
num_vertice = int( lines[0] )
num_line += 1
vertice = [0 for _ in range( num_vertice + 1 ) ]
for i in range( num_line, num_line + num_vertice ):
  vertice[i - num_line] = int( lines[i].split()[0] )
num_line += num_vertice

root = int( lines[num_line] )
num_line += 1
num_edge = int( lines[num_line] )
vertice[ num_vertice ] = num_edge
num_line += 1
print( "#vertice: ", num_vertice )
print( "#edge   : ", num_edge )
print( "root    : ", root )

edge = [0 for _ in range( num_edge ) ]
for i in range( num_line, num_line + num_edge ):
  edge[i-num_line] = int( lines[i].split()[0] )
num_line += num_vertice

visit = [num_vertice for _ in range( num_vertice ) ]
level = 1
visit[ root ] = 0
next_vertice = []
next_level = []
next_vertice.append( root )
next_level.append( 1 )
while len( next_vertice ) > 0:
  current = next_vertice.pop(0)
  level = next_level.pop(0)
  for i in range( vertice[ current ], vertice[ current + 1 ] ):
    if visit[edge[i]] > level:
      visit[edge[i]] = level
      next_vertice.append( edge[ i ] )
      next_level.append( level + 1 )

print( "vertice : ", vertice )
print( "edge    : ", edge )
print( "visit   : ", visit )
