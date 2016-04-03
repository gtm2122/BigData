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
def py_trans_time(a,b):
	start = time.time()
	b = np.transpose(a)
	return time.time()-start 
def py_trans(a):
       	return np.transpose(a)
	
def create_arrays(P,Q):
    a = (np.random.rand(P, Q).astype(np.float32))
    c = np.empty((Q, P), dtype=np.float32 )
    return a, c
###transpose using tiling with square matrices
func = cl.Program(ctx, """ __kernel void func(__global float* a, 
__global float* b, const unsigned int m, const unsigned int n) {
	__local float ds_A[2][2];
	
	const int tx=get_local_id(0);
	const int ty = get_local_id(1);
	const int gid0 = 2*get_group_id(0)+tx;
	const int gid1 = 2*get_group_id(1)+ty;
	
	if (gid0< m && gid1 < n)
	{
	ds_A[ty][tx] = a[gid1*m + gid0];
	}
	else
	{ds_A[ty][tx] = 0.0;}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	const int gid0_t = 2*get_group_id(1) + tx;
        const int gid1_t = 2*get_group_id(0) + ty;
	if (gid0_t < n && gid1_t < m)
		{b[gid1_t*n + gid0_t]=ds_A[tx][ty];}
}	
""").build().func 
func.set_scalar_arg_dtypes([None, None,
                             np.uint32, np.uint32])
def op_trans(a_gpu,b_gpu,m,n):
	start = time.time()
	func(queue, (n,m) , (2,2), a_gpu.data,b_gpu.data,m,n)
	return time.time() - start 
	
def to_gpu(a,b,m,n):
	a_gpu = cl.array.to_device(queue,a)
	b_gpu = cl.array.empty(queue,(n,m),a.dtype)
	return a_gpu,b_gpu 
l = 6 
m = 6 
n = 8 
a,b = create_arrays(m,n) 
d,c = create_arrays(l,m) 
a_gpu,b_gpu = to_gpu(a,b,m,n) 
d_gpu,c_gpu = to_gpu(d,c,l,m) 
op_trans(d_gpu,c_gpu,l,m) 
d_trans=py_trans(d) 
c_op = c_gpu.get() 
print 'd = ',d 
print 'python transpose is ',d_trans 
print 'opencl transpose using tiling is ',c_op 
print ' equal ', np.allclose(d_trans,c_op) 
 
 
import csv 
import matplotlib as mpl 
mpl.use('agg') 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.cm as cm 
mpl.rcParams['savefig.dpi']=100 
print ' graphs for transpose using tiling' 
l = 4 
m = 4
	
py_times=[] 
op_times=[] 
dim_mat=[] 
for x in range (1,50):
	a,b=create_arrays(l,m)
	a_gpu,b_gpu=to_gpu(a,b,l,m)
	py_times.append(py_trans_time(a,b))
	op_times.append(op_trans(a_gpu,b_gpu,l,m))
	dim_mat.append(l*m)
	l = l+2
	m = m+2
	
print ' dim_mat_tiling_trans',dim_mat 
print ' py_times_tiling_trans',py_times 
print ' op_times_tiling_trans',op_times 
plt.clf() 
plt.plot(dim_mat,py_times,'bo-',dim_mat,op_times,'ro-') 
plt.xlabel('N X P in ratio 1 : 2 )') 
plt.ylabel('time') 
plt.title('Time comparision while using tiling') 
plt.legend(('Python','PyOpenCL')) 
plt.grid(True) 
plt.gca().set_xlim((min(dim_mat),max(dim_mat))) 
plt.draw() 
plt.savefig('tiling_trans.png') 
