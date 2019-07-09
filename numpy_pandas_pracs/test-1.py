import numpy as np
# import cupy as cp
import cupy
#print("123")

'''
x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)
#f'l2_gpu==> {l2_gpu}'
#print("l2_gpu==>", l2_gpu)
print(f'l2_gpu==> {l2_gpu}')

x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)
print(f'l2_cpu==> {l2_cpu}')
'''

'''
with cp.cuda.Device(0):
#with cp.cuda.Device(1):
    x = cp.array([1, 2, 3, 4, 5])
print(x.device)
'''

'''
x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)
print(x_gpu)
'''

'''
squared_diff = cp.ElementwiseKernel(
'float32 x, float32 y',
'float32 z',
'z = (x - y) * (x - y)',
'squared_diff')
'''

'''
x = cp.arange(10, dtype=np.float32).reshape(2, 5)
print("x")
print(x)
print("==================")


y = cp.arange(5, dtype=np.float32)
print("y")
print(y)



print("==================")
print(squared_diff(x, y))
print("==================")
#print(squared_diff(x, 5))
print(squared_diff(x, 5).data)
print("==================")


z = cp.empty((2, 5), dtype=np.float32)
#z = cp.empty((10, 5), dtype=np.float32)
#z = cp.empty((2, 5))
print("z")
print(z.data)
print(z)

z2 = np.empty((2, 5), dtype=np.float32)
print("z2")
print(z2)
print(z2.data)
'''

'''
l2norm_kernel = cp.ReductionKernel(
'T x',  # input params
'T y',  # output params
'x * x',  # map
'a + b',  # reduce
'y = sqrt(a)',  # post-reduction map
'0',  # identity value
'l2norm'  # kernel name
)

x = cp.arange(10, dtype=np.float32).reshape(2, 5)
#print(l2norm(x, axis=1))
print(l2norm_kernel(x, axis=1))
#y = l2norm_kernel(x, axis=1)
#print(y)
'''

# '''
# add_kernel = cp.RawKernel(r'''
# extern "C" __global__
# void my_add(const float* x1, const float* x2, float* y) {
#     int tid = blockDim.x * blockIdx.x + threadIdx.x;
#     y[tid] = x1[tid] + x2[tid];
# }
# ''', 'my_add')

'''
x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
print(x1)

x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
print(x2)

y = cp.zeros((5, 5), dtype=cp.float32)

#z=add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
print(y)
#print(z)
'''

# @cp.fuse()
# def squared_diff(x, y):
#     return (x - y) * (x - y)
# x_cp = cp.arange(10)
# print(x_cp)
# y_cp = cp.arange(10)[::-1]
# print(y_cp)
# print(squared_diff(x_cp, y_cp))
#
# x_np = np.arange(10)
# print(x_np)
# y_np = np.arange(10)[::-1]
# print(y_np)
# print(squared_diff(x_np, y_np))

# a = cupy.arange(3)
# print(a)
# print(a[[1, 3]])
# print(a[[1, 2]])
# print(a[[1, 0]])

# print(np.array([-1], dtype=np.float32).astype(np.uint32))
# print(cupy.array([-1], dtype=np.float32).astype(np.uint32))



# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)


























































