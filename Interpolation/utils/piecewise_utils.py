"""
powered by: @御河DE天街
计算插值和绘图的工具包
只适用于插值多项式为分段函数的情况
如果插值多项式是单一表达式则无法使用
"""

def cal_interp(polynomial,x,x0):
    """
    计算给定插值点的数值，即插值
    polynomial:插值多项式
    x0:所求插值的x坐标值
    x:插值点的x坐标值
    """
    
    import numpy as np
    import sympy as sp
    
    x0=np.asarray(x0,dtype=np.float64)
    n0=len(x0)      #所求插值点的个数
    y_0=np.zeros(n0)    #储存插值点y0所对应的插值
    t = polynomial[0].free_symbols.pop()    #获取多项式的自由变量
    for i in range(n0):
        idx=0   #子区间索引值初始化
        for j in range(len(x)-1):
            #查找x0所在的子区间，获取子区间的索引值idx
            if x[j]<=x0[i]<=x[j+1] or x[j+1]<=x0[i]<=x[j]:
                idx=j
                break
        y_0[i]=polynomial[idx].evalf(subs={t:x0[i]})
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
    yi=cal_interp(polynomial,x,xi)
    plt.plot(xi,yi,'b--',label='Interpolation')
    if x0 is not None and y0 is not None:
        plt.plot(x0,y0,'g*',label='Cal points')
    plt.legend()
    plt.xlabel('x',fontdict={'fontsize':12})
    plt.ylabel('y',fontdict={'fontsize':12})
    plt.title(title + ' Interpolation',fontdict={'fontsize':14})
    plt.grid(linestyle=':')
    plt.show() 
