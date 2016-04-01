import csv
import numpy as np
import time
import numpy.matlib
import pyopencl as cl
from numpy.linalg import inv
X = np.genfromtxt("./AAPL.txt",delimiter=',',dtype='float')
#print np.shape(X)
days = np.zeros((250,1)).astype(np.float32)
for i in range(250):
        days[i,:]=i+1
#print days
Xtrain=np.zeros((250,3)).astype(np.float32)

Xtrain[:,0] = days[0:250,0]
Xtrain[:,1] = X[0:250,4]
Xtrain[:,2] = X[0:250,5]
#print (Xtrain[:,2])
Ytrain = np.zeros((250,1)).astype(np.float32)
Ytrain = X[0:250,1]
Xtest = np.zeros((1,3)).astype(np.float32)
Y_251 = 109.56
Xtest = [251 , 108.65, 45159900]
#print np.shape(np.transpose(Xtrain))
#print np.shape(Ytrain)
#print np.dot(np.transpose(Xtrain),Ytrain)
t = time.time()
W_ser =np.dot(inv(np.dot(Xtrain.T,Xtrain)),np.dot(Xtrain.T,Ytrain))
Y_pred_ser =np.dot(Xtest,W_ser)
print(Y_pred_ser)
print time.time() - t

L=3
M=250
N=3

blocks=10

NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
        if platform.name == NAME:
                devs = platform.get_devices()

# Setting Command Queue.
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)


kernel2 = """
#include <pyopencl-complex.h>
__kernel void matmul3(__global cfloat_t* X, __global cfloat_t* C, __global cfloat_t* Y , const int L, const int M, const int N) {
	
	unsigned int Col, k;
	cfloat_t Awrk[1024];
	unsigned int Row = get_global_id(0);
	for (k = 0; k < M; k++)
	{
		Awrk[k] = X[Row*M + k];
	}
	for (Col = 0; Col < N ; Col++)
	{
		cfloat_t temp = 0;
		for (k=0; k < M ; k++)
		{
			temp = temp + cfloat_mul(Awrk[k], C[k*N + Col]);
//			temp = 1.0;
		}
		Y[Row*N + Col] = temp;
	}
}
"""

kernel3 = """
__kernel void algo1(__global float* A, __global float* B, __global float* output, const int L, const int M, const int N) {
	
	float Awrk[1024]; // We define private memory
	unsigned int row = get_global_id(0); 
	unsigned int icol = get_local_id(0);
	unsigned int nCol = get_local_size(0); 
	unsigned int col, k;
	__local float Bloc[1024];	
	
	for (int k = 0; k < M; k++)
	{
		Awrk[k] = A[row*M + k];
	}
	for (int col = 0; col < N ; col++)
	{	
		
		for (int k = icol; k < M; k += nCol)
		{
			Bloc[k] = B[k*N + col] ;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		float temp = 0;
		for (int k=0; k < M ; k++)
		{
			temp = temp + Awrk[k]*Bloc[k];
		}
		output[row*N + col] = temp;
	}
}
"""


c3 = np.zeros((3,3)).astype(np.float32)
#Defining buffers and clearing the memory flags
mf = cl.mem_flags
x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xtrain.T)
c_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xtrain)

c3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c3.nbytes)


#Building the kernel and passing values to the kernel followed by getting a copy of the output from the GPU
prg = cl.Program(ctx, kernel3).build()
t = time.time()
prg.algo1(queue, (L, ),(10,), x_buf, c_buf, c3_buf, np.int32(L), np.int32(M), np.int32(N))
cl.enqueue_copy(queue, c3, c3_buf)
print time.time() - t
#Kernel build ,calling the appropriate kernel function followed by getting the values from the GPU


print c3
print np.dot(Xtrain.T,Xtrain)
#W_par =np.dot(np.dot(out,Xtrain),y)

#y_par_pred = np.dot(Xtest,W_par)

print np.allclose(c3,inv(np.dot(Xtrain.T,Xtrain)))
