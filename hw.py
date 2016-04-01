
import csv
import time 
import pyopencl as cl 
import pyopencl.array 
import numpy as np 
NAME = 'NVIDIA CUDA' 
platforms = cl.get_platforms() 
devs = None 
for platform in platforms:
	if platform.name == NAME:
		devs = platform.get_devices() 
ctx = cl.Context(devs[0:1]) 
queue = cl.CommandQueue(ctx) 
def mult_py(a,b,c,d):
	start = time.time()
	d[:,:] = np.dot(a,(b+c))
	return d

	
def create_arrays(P,Q,R):
	a=np.random.rand(P,Q).astype(np.float32)
	b=np.random.rand(Q,R).astype(np.float32)
	c=np.random.rand(Q,R).astype(np.float32)
	d=np.zeros((P,R),dtype=np.float32)
	return a,b,c,d 
def to_gpu(a,b,c):
	a_gpu = cl.array.to_device(queue,a)
	b_gpu = cl.array.to_device(queue,b)
	c_gpu = cl.array.to_device(queue,c)
	d_gpu = cl.array.empty(queue,(a_gpu.shape),a.dtype)
	return a_gpu,b_gpu,c_gpu,d_gpu 
P = 4 
Q = 4 
R = 4 
a,b,c,d=create_arrays(P,Q,R) 
a_gpu,b_gpu,c_gpu,d_gpu = to_gpu(a,b,c) 
d = mult_py(a,b,c,d)

func1 = cl.Program(ctx, """ 
__kernel void func(__global const float* a, __global const float* b, __global const float* c, __global float* d, const int P,const int Q,const int R) 
{ __local float dsa[2][2];
 __local float dsb[2][2];
 unsigned int z = get_global_id(0);
 unsigned int z1 = get_global_id(1);
 //b[z*R+ z1] = b[z*R+z1] + c[z*R+z1]; 
unsigned int bx = get_group_id(0); 
unsigned int by = get_group_id(1); 
unsigned int tx = get_local_id(0); 
unsigned int ty = get_local_id(1); 
unsigned int Row = by*2+ty; 
unsigned int Col = bx*2 + tx; 
float cvalue=0.0; 
for ( int t = 0 ; t<P/2;++t) {
	//if ( Row < P && t*2+tx < Q)
	{
	dsa[ty][tx]=a[Row*Q+t*2+tx];
	}
//	else
	{
	// dsa[ty][tx]=0.0;
	}
	//if (t*2+ty < Q && Col <R)
	{
	dsb[ty][tx] = b[(t*2+ty)*R + Col]+c[(t*2+ty)*R+Col];
	}
	//else
	
	{
	//dsb[ty][tx]=0.0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = 0 ; i <2 ; ++i)
	{
	cvalue+=dsa[ty][i]*dsb[i][tx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}	
//	if(Row<P && Col<R)
	d[Row*R+Col] = cvalue;
}
""").build().func 
func1.set_scalar_arg_dtypes([None,None,None,None,np.uint32,np.uint32,np.uint32]) 
a,b,c,d=create_arrays(P,Q,R)

print 'prev d_op1 =',d 
a_gpu,b_gpu,c_gpu,d_gpu=to_gpu(a,b,c) 
def mult_op1(a_gpu,b_gpu,c_gpu,d_gpu):
	start = time.time()
	P,Q = a_gpu.shape
	R = b_gpu.shape[1]
	func1(queue,(P,R),(2,2),a_gpu.data,b_gpu.data,c_gpu.data,d_gpu.data,P,Q,R)
	return np.log(time.time()-start) 
print ' dop ',d_gpu.get() 
d=mult_py(a,b,c,d) 
mult_op1(a_gpu,b_gpu,c_gpu,d_gpu) 
d_op1 = d_gpu.get() 
print ' a = ',a 
print ' b+c=',b+c 
print 'd = ',d 
print 'd_op1 = ',d_op1 
print 'true',np.allclose(d,d_op1)

import csv 
import matplotlib as mpl 
mpl.use('agg') 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.cm as cm 
mpl.rcParams['savefig.dpi']=100 
print ' graph for Naive vs tiled ' 
Ntemp=4 
Ptemp=4 
Mtemp=4 
py_times=[] 
op1_times=[] 
dim_mat=[] 
for x in range (1,12):
	a,b,c,d=create_arrays(Ntemp,Ptemp,Mtemp)
	a_gpu,b_gpu,c_gpu,d_gpu=to_gpu(a,b,c)
	op1_times.append(mult_op1(a_gpu,b_gpu,c_gpu,d_gpu))
	t=time.time()
	g=np.dot(a,b+c)
	t=np.log(time.time()-t)
	py_times.append(t)
	dim_mat.append(Ntemp*Ptemp*Mtemp)
	Ntemp=2*Ntemp
	Mtemp=2*Mtemp
	Ptemp = 2*Ptemp
	
print ' dim_mat',dim_mat 
print ' tiling times on pyopencl',op1_times 
plt.clf() 
plt.plot(dim_mat,py_times,'bo-',dim_mat,op1_times,'ro-') 
plt.xlabel('m*n*p') 
plt.ylabel('time') 
plt.title('Time comparision while using python numpy and tiled implementation in opencl') 
plt.legend(('python','tiled')) 
plt.grid(True) 
plt.gca().set_xlim((min(dim_mat),max(dim_mat))) 
plt.draw() 
plt.savefig('unop_np.png')
