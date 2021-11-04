import taichi as ti 
import numpy as np
from PIL import Image, ImageDraw

ti.init(arch=ti.gpu)
eps  = 1e-4
e_ = ti.Vector([1,0])
#canvas size 
n = 320
pixels = ti.Vector.field(4, ti.f32, shape=(n * 2, n))
root_memo = [] 
zoom_ = 1
@ti.func
def complex_sqr(z)->ti.Vector:
	return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.func
def complex_mul(x, y)->ti.Vector:
    return ti.Vector([x[0]*y[0] - y[1]*x[1], x[1] * y[0] + y[1]*x[0]])

@ti.func
def ndegreeFunc(z,n):
    t = z 
    for _ in range(n-1):
        z = complex_mul(z,t )
    return z 

@ti.func
def func(z,n):
    val = ndegreeFunc(z,n)
    return ti.Vector([val[0]- 1, val[1]])

@ti.func
def complex_div(x,y)->ti.Vector:
    return ti.Vector(
       [ 
           (x[0]*y[0] + x[1]*y[1])/(y.norm()**2 + 1e-3), 
           (-x[0]*y[1]+x[1]*y[0])/(y.norm()**2 + 1e-3)
       ]
    )
@ti.func
def color_render(ang):
    """Color Rendering
    Input Ang return Color on the position 
    Args:
        ang ([type]): [description]
    """
    return ti.abs(ti.Vector([ti.cos(ang), ti.cos(ang + 0.3), ti.cos(ang + 0.6), 0.5]))

@ti.kernel
def calc_test(N:int,t:float,zoom_:float):
    threshold = 1000
    for i,j in pixels:

        col = (i - 640 / 2 + 0.5) / zoom_
        row = (j - 320 / 2 + 0.5) / zoom_

        x = ti.Vector([col,row])

        time_rotate_mat = ti.Vector([
            [ti.cos(t),ti.sin(t)],
            [-ti.sin(t),ti.cos(t)]
            ])
        
        x = time_rotate_mat @ x
        fv = func(x,N)
        gv = 6 * ndegreeFunc(x,N-1)

        div= complex_div(fv,gv)
        iter_num = 0
        
        # print("====Iter:{}======".format(0))
        # print("diff: {}".format((x-div).norm()))

        while div.norm()> eps and iter_num <threshold:
            
            x = x-div
            
            fv = func(x,N)
            gv = 6 * ndegreeFunc(x,N-1)
            div= complex_div(fv,gv)

            # if iter_num%50 == 0:
            #     print("====Iter:{}======".format(iter_num))
            #     print("x:[{},{}]".format(x[0], x[1]))
            #     print("fv:[{},{}]".format(fv[0], fv[1]))
            #     print("gv:[{},{}]".format(gv[0], gv[1]))
            #     print("div:[{},{}]".format(div[0], div[1]))
            #     print("diff: {}".format((x-div).norm()))
                
            iter_num +=1
        #final pos
        x = x-div
        ang = ti.atan2(x[1],x[0])
        pixels[i, j] = color_render(ang)

# gui = ti.GUI("Newton Set", res=(n * 2, n))

zoom_rate = 1
images = []
for i in range(1000):
    calc_test(3, i * 0.01,zoom_ * zoom_rate)
    img = pixels.to_numpy() *255
    img = img.astype(np.uint8)
    images.append(Image.fromarray(img[:,:,:3]))
    zoom_rate = 1 + i*0.01 
    # gui.set_image(pixels)
    # gui.show()

images[0].save(f'data/newton_fractal.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=1)