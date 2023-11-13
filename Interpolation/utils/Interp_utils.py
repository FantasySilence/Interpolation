"""
powered by: @御河DE天街
计算插值和绘图的工具包
只适用于插值多项式为单一多项式的情况
如果插值多项式是分段函数则无法使用
"""

def cal_interp(polynomial,x0):
    """
    计算给定插值点的数值
    x0:所求插值的x坐标值
    polynomial:插值多项式
    """
    
    import numpy as np

    x0=np.asarray(x0,dtype=np.float32)
    n0=len(x0)  #所求插值点的数量
    y_0=np.zeros(n0)    #储存插值点x0所对应的插值
    t=polynomial.free_symbols.pop()  #获取多项式中的符号变量 # type: ignore
    for i in range(n0):
        y_0[i]=polynomial.evalf(subs={t:x0[i]}) # type: ignore
    return y_0

def plt_interp(params):
    """
    可视化插值图像和插值点
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    polynomial,x,y,title,x0,y0 = params
    plt.figure(figsize=(8,6))
    plt.plot(x,y,'ro',label='Interp points')
    xi=np.linspace(min(x),max(x),100)
    yi=cal_interp(polynomial,xi)
    plt.plot(xi,yi,'b--',label='Interpolation')
    if x0 is not None and y0 is not None:
        plt.plot(x0,y0,'g*',label='Cal points')
    plt.legend()
    plt.xlabel('x',fontdict={'fontsize':12})
    plt.ylabel('y',fontdict={'fontsize':12})
    plt.title(title + ' Interpolation',fontdict={'fontsize':14})
    plt.grid(linestyle=':')
    plt.show()