import sys
import numpy as np

n=5000

random_matrix = np.random.rand(n,n);

mat = np.matrix(random_matrix)

print(mat)

with open('input_'+str(n)+'_'+str(n)+'.txt','w') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.12f')