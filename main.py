import taichi as ti
import numpy as np

from taichi.lang.ops import sqrt 
ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

eps = 1e-4
memo_rt = []


@ti.func
def complex_sqr(z)->ti.Vector:
	return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.func
def ndegreeFunc(z,n):
    for _ in range(n-1):
        z = complex_sqr(z)
    return z 

@ti.func
def func(z,n):
    val = ndegreeFunc(z,n)
    return ti.Vector([val[0]- 1, val[1]])

@ti.func
def complex_func(z)->ti.Vector:
	return ti.Vector([z[0]**3 - 3*z[0]*z[1]**2 -1, -z[1]**3 + 3*z[0] * z[1] ** 2])

@ti.func
def complex_div(x,y)->ti.Vector:
    return ti.Vector(
       [ 
           (x[0]*y[0] + x[1]*y[1])/(y.norm()**2 + 1e-3), 
           (-x[0]*y[1]+x[1]*y[0])/(y.norm()**2 + 1e-3)
       ]
    )
@ti.kernel
def main(t:float):
    for i,j in pixels:
        iter_num = 0
        z = ti.Vector([i / n - 1  , j / n - 0.5]) * 5
        time_rotate_mat = ti.Vector([
            [ti.cos(t),ti.sin(t)],
            [-ti.sin(t),ti.cos(t)]
            ])
        
        z = time_rotate_mat @ z
        threshold = 1000
        func_val = func(z,6)
        grad_val = ndegreeFunc(z,5)

        diff = z - complex_div(func_val, grad_val)

        while diff.norm() > 1e-4 and iter_num < threshold:
            z = diff
            func_val = func(z,6)
            grad_val = ndegreeFunc(z,5)

            diff = z - complex_div(func_val, grad_val)
            iter_num+=1

        z= diff
        print(z.norm()!= z.norm())
        pixels[i, j] = 1 - z.norm()/(5*sqrt(2))

gui = ti.GUI("Newton Set", res=(n * 2, n))

for i in range(1000000):
    main(i * 0.03)
    gui.set_image(pixels)
    gui.show()
