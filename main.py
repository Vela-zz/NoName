import taichi as ti
import numpy as np 
ti.init(arch=ti.gpu)

n = 320
pixels = ti.field(dtype=float, shape=(n * 2, n))

eps = 1e-4
memo_rt = []


@ti.func
def complex_sqr(z)->ti.Vector:
	return ti.Vector([3 *z[0]**2 - z[1]**2, 3*z[1] * z[0] * 2])

@ti.func
def complex_func(z)->ti.Vector:
	return ti.Vector([z[0]**3 - 3*z[0]*z[1]**2 -1, -z[1]**3 + 3*z[0] * z[1] ** 2])

@ti.func
def complex_div(x,y):
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
        z = ti.Vector([i / n - 1 + ti.cos(t)*0.02 , j / n - 0.5 + ti.sin(t) *0.02]) * 5
        threshold = 1000
        func_val = complex_func(z)
        grad_val = complex_sqr(z)

        nxt_pos = z - complex_div(func_val, grad_val)

        while (z-nxt_pos).norm() > 1e-4 and iter_num < threshold:
            z = nxt_pos
            func_val = complex_func(z)
            if func_val.norm()>1e10:
                break
            grad_val = complex_sqr(z)
            nxt_pos = z - complex_div(func_val, grad_val)

            iter_num+=1
        #solution
        z= nxt_pos
        # isExsit =False
        # ind = 0
        # if z.all():
        #     for r in memo_rt:
        #         if (r-z).norm()<1e-4:
        #             isExsit = True
        #             z = r
        #             break
        #         ind +=1
            
        #     if not isExsit:
        #         memo_rt.append(z)
        #         ind= len(memo_rt)

        pixels[i, j] = 1 - z.norm() *0.02

gui = ti.GUI("Newton Set", res=(n * 2, n))

for i in range(1000000):
    main(i * 0.03)
    gui.set_image(pixels)
    gui.show()
