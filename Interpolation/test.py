import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
from CubicSpline import CubicSplineInterpolation
from Lagrange import LagrangeInterpolation
from piecewise_utils import cal_interp

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 将三种边界条件的三次样条插值绘制于同一图表中
def plt_interp(params):
    """
    可视化插值图像和插值点
    """
    polynomial,x,y,title,x0,y0 = params
    plt.plot(x,y,'ro',label='Interp points')
    xi=np.linspace(min(x),max(x),100)
    yi=cal_interp(polynomial,x,xi)
    plt.plot(xi,yi,'b--',label='Interpolation')
    if x0 is not None and y0 is not None:
        plt.plot(x0,y0,'g*',label='Cal points')
    plt.legend()
    plt.xlabel('x',fontdict={'fontsize':12})
    plt.ylabel('y',fontdict={'fontsize':12})
    plt.title(title + ' Interpolation',fontdict={'fontsize':14})
    plt.grid(linestyle=':')

if __name__ == "__main__":

    # x=np.linspace(0,2*np.pi,10,endpoint=True)
    # y=np.sin(x)
    # dy=np.cos(x)
    # d2y=-np.sin(x)
    # x0=np.array([np.pi/2,2.158,3.58,4.784])

    x = np.linspace(0,2*np.pi,10)
    y = 2*np.exp(-x)*np.sin(x)
    dy = 2*np.exp(-x)*(np.cos(x)-np.sin(x))
    d2y = 2*np.exp(-x)*(-2*np.cos(x)+np.sin(x))
    x0=np.array([np.pi/2,2.158,3.58,4.784])

    bc=['complete','second','natural','periodic']
    plt.figure(figsize=(14,10))
    sp=0
    for bc_ in bc:
        csi_Interp=CubicSplineInterpolation(x,y,dy=dy,d2y=d2y,boundary_type=bc_)
        csi_Interp.fit_interp()
        y0=csi_Interp.cal_interp(x0)
        print('插值点的值为：',y0)
        params=(csi_Interp.polynomial,csi_Interp.x,csi_Interp.y,"Cubic Spline(%s)"% bc_,x0,y0)
        plt.subplot(221+sp)
        plt_interp(params) 
        sp+=1
    plt.show()

# 与原函数比较
def fun(x):
    return 1/(1+25*x**2)

if __name__ == '__main__':
    x=np.linspace(-1,1,11,endpoint=True)
    y=fun(x)
    x0=np.linspace(-1,1,100,endpoint=True)
    lag_Interp=LagrangeInterpolation(x=x,y=y)
    lag_Interp.fit_interp()
    y1=lag_Interp.cal_interp(x0)
    plt.figure(figsize=(8,6))
    plt.title('Lagrange Interpolation')
    plt.plot(x0,y1,'b--',label='Lagrange Interpolation')
    plt.plot(x,y,'ro',label='original data')
    plt.plot(x0,fun(x0),'g-',label='fitting line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()