import numpy as np
import cupy
import pandas as pd

# pd.test()

# array_a = np.array([[1,2,3],
#  [2,3,4]])
#
# #print(array_a)
# print("array:", array_a)
# print("ndim:", array_a.ndim)
# print("shape:", array_a.shape)
# print("size:", array_a.size)

# a=np.array(([1,2,3]), dtype=np.int64)
# # b=np.array(([1,2,3]), dtype=int)
# b=np.array(([1,2,3]), dtype=np.int32)
# c=np.array(([1,2,3]), dtype=np.float)
# #d=np.zeros((3,5), dtype=np.float)
# #d=np.empty((3,5), dtype=np.int64)
# #d=np.empty((3,5), dtype=np.float)
# d=np.empty((3,5))
# print(a.dtype)
# print(b.dtype)
# print(c.dtype)
# print(d)

# #a = np.arange(1,100,dtype=np.int64)
# # a = np.arange(12).reshape(3,4)
# a = np.arange(1, 200, 2).reshape(10,10)
# b = np.arange(1, 450.0, 4.5, dtype = float).reshape(10,10)
# print(a)
# print("shape:", a.shape)
# print(b)
# print("shape:", b.shape)
# c=20*np.sin(a) - b
# #print(c)
# # print(np.abs(c))
# print(np.ceil(c))
# # print(c<-150)
# # print(np.abs(c)==-150)
# print(np.ceil(c)==-201)

# import cupy
# import numpy as np
# import pandas as pd
#
# add_kernel = cupy.RawKernel(r'''
# extern "C" __global__
# void my_add(const float* x1, const float* x2, float* y) {
#      int tid = blockDim.x * blockIdx.x + threadIdx.x;
#      y[tid] = x1[tid] + x2[tid];
# }
# ''', 'my_add')
#
# x1 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
# x2 = cupy.arange(25, dtype=cupy.float32).reshape(5, 5)
# y = cupy.zeros((5, 5), dtype=cupy.float32)
# add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
# print(y)

# c=2
# print(c**2*c)

# a = np.array([[1,1], [0,1]])
# print(a)
# b=np.arange(4).reshape(2,2)
# print(b)
# c=a*b
# print(c)
# d_dot=np.dot(a,b)
# print(d_dot)
# e_dot=a.dot(b)
# print(e_dot)


# a = np.random.random((3,6))
# print(a)
# print(np.sum(a))
# print(np.max(a))
# print(np.min(a))
# print(np.max(a, axis=0))
# print(np.sum(a, axis=1))
# print(np.sum(a, axis=-1))

# a = np.arange(3,18).reshape((5,3))
# b = np.arange(18,3,-1).reshape((5,3))
# c = np.arange(18,2,-1).reshape((4,4))
# c = np.random.random((5,5))
# d = np.arange(93,3,-1).reshape((9,10))
# print(a)
# print(np.argmin(a)) #index position
# print(np.argmax(a)) #index position
# print(np.mean(a))
# print(a.mean())
# print(np.average(a))
# #print(a.median)
# print(np.median(a))
# print(np.sum(a))
# print(np.cumsum(a))
# print(np.nansum(a))
# print(np.diff(a))
# print(np.nonzero(a))

# print(b)
# print(np.sort(b))
# print(np.transpose(b))
# print(b.T)
# print(np.sort(b.T))

# ###print((b.T)**-1)
# # c_T = c.T
# # print(np.linalg.inv(c.T))
# print(np.linalg.inv(c.T))
# ###print(((b)**(-1)).T)
# #print(np.linalg.inv(c).T)
# print(np.linalg.inv(c).T)

# print(np.clip(d, 15, 63))
# print(np.average(d, axis=1))
# print(np.mean(d, axis=0))

# a = np.arange(4,32)
# print(a)
# print(a[5])

# a = np.arange(4,32).reshape(7,4)
# print(a)
# print(a[2][2])
# print(a[0][:])
# print(a[0,:])

# for row in a:
#     print(row)

# for column in a.T:
#     print(column)

# print(a.flat)
# for item in a.flat:
#     print(item)

# print(a.flatten())

# a = np.array([1,1,1])
# b = np.array([3,3,3])
# # print(np.vstack((a,b))) #vertical
# c = np.vstack((a,b))
# print(c)
# print(a.shape,c.shape)
# d = np.hstack((a,b))
# print(d)
# print(a.shape,d.shape)
# # print(a.T)
# print(a[:, np.newaxis])
# a = np.array([4,4,4])[:, np.newaxis]
# b = np.array([7,7,7])[:, np.newaxis]
# e=np.hstack((a,b))
# print(e)

# a = np.array([4,4,4])[:, np.newaxis]
# b = np.array([7,7,7])[:, np.newaxis]
# c=np.concatenate((a,b,b,a), axis=1)
# print(c)

# a = np.arange(144).reshape(12,12)
# print(a)
# # b = np.split(a, 12, axis=0)
# # print(b)
# c = np.split(a, 12, axis=1)
# print(c)

# a = cupy.arange(12).reshape(4,3)
# print(a)
# print(cupy.split(a,3, axis =1))
# print(cupy.split(a,2, axis =0))
# print(cupy.split(a,4, axis =0))
# print(cupy.array_split(a,6, axis =0))
# print(cupy.array_split(a,3, axis =0))
# print(cupy.vsplit(a,4))
# print(cupy.hsplit(a,3))

# a = cupy.arange(8)
# b=a
# c=a
# d=b
# a[0]=98
# print(a)
# print(b)
# print(c)
# print(d)
# print(d is a)
# e=a.copy()
# print(e)
# a[4]=55
# print(a)
# e[3]= 99
# print(e)

