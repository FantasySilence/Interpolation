import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class CubicSplineInterpolation:
    """
    三次样条插值
    complete：第一种边界条件，已知两端的一阶导数值
    second：第二种边界条件，已知两端的二阶导数值
    natural：第二种边界条件，自然边界条件
    periodic：第三种边界条件，周期边界条件
    """
    
    def __init__(self, x, y,dy=None,d2y=None,boundary_type='natural'):
        """
        三次样条插值必要参数的初始化
        """
        self.x=np.asarray(x,dtype=np.float64)
        self.y=np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)      #已知离散数据点的个数
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.dy,self.d2y=dy,d2y     #边界条件的一阶导数值和二阶导数值
        self.boundary_type=boundary_type   #边界条件的类型,默认为自然边界条件
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从高到低
        self.y0=None   #所求插值点的值，单个值或者向量
    
    def fit_interp(self):
        """
        构造三次样条插值多项式
        """
        t=sp.Symbol('t')
        self.polynomial=dict()
        self.poly_coefficient=np.zeros((self.n-1,4))
        if self.boundary_type == "complete":     #给定边界一阶导数相等
            if self.dy is not None:
                self.dy=np.asarray(self.dy,dtype=np.float64)
                self.__complete_spline__(t,self.x,self.y,self.dy)
            else:
                raise ValueError("第一种边界条件，需要给定边界处的一阶导数值")
        elif self.boundary_type == "second":     #给定边界二阶导数相等
            if self.d2y is not None:
                self.d2y=np.asarray(self.d2y,dtype=np.float64)
                self.__second_spline__(t,self.x,self.y,self.d2y)
            else:
                raise ValueError("第二种边界条件，需要给定边界处的二阶导数值")
        elif self.boundary_type == "natural":     #自然边界条件
            self.__natural_spline__(t,self.x,self.y)
        elif self.boundary_type == "periodic":     #周期边界条件
            self.__periodic_spline__(t,self.x,self.y)
        else:
            raise ValueError("边界条件类型错误！")
    
    def __spline_poly__(self,t,x,y,m):
        """
        三次样条插值多项式的构造
        t:符号变量
        x:已知数据的x坐标点
        m:求解的矩阵系数，即多项式m的值
        """
        for i in range(self.n-1):
            hi=x[i+1]-x[i]      #子区间长度
            pi=y[i]/hi**3*(2*(t-x[i])+hi)*(x[i+1]-t)**2+\
                y[i+1]/hi**3*(2*(x[i+1]-t)+hi)*(t-x[i])**2+\
                    m[i]/hi**2*(t-x[i])*(x[i+1]-t)**2-\
                        m[i+1]/hi**2*(x[i+1]-t)*(t-x[i])**2
            self.polynomial[i]=sp.simplify(pi)
            poly_obj=sp.Poly(pi,t)
            mons=poly_obj.monoms()
            for j in range(len(mons)):
                self.poly_coefficient[i,mons[j][0]]=poly_obj.coeffs()[j]
    
    def __complete_spline__(self,t,x,y,dy):
        """
        第一种边界条件的三次样条插值
        """
        A=np.diag(2*np.ones(self.n))
        c=np.zeros(self.n)
        for i in range(1,self.n-1):
            u=(x[i]-x[i-1])/(x[i+1]-x[i-1])
            lambda_=(x[i+1]-x[i])/(x[i+1]-x[i-1])
            c[i]=3*lambda_*(y[i]-y[i-1])/(x[i]-x[i-1])+\
                 3*u*(y[i+1]-y[i])/(x[i+1]-x[i])
            A[i,i+1],A[i,i-1]=u,lambda_
        c[0],c[-1]=2*dy[0],2*dy[-1]     #边界条件
        m=np.linalg.solve(A,c)
        self.__spline_poly__(t,x,y,m)     #构造三次样条插值多项式

    def __second_spline__(self,t,x,y,d2y):
        """
        第二种边界条件的三次样条插值
        """
        A=np.diag(2*np.ones(self.n))
        A[0,1],A[-1,-2]=1,1     #边界特殊情况
        c=np.zeros(self.n)
        for i in range(1,self.n-1):
            u=(x[i]-x[i-1])/(x[i+1]-x[i-1])
            lambda_=(x[i+1]-x[i])/(x[i+1]-x[i-1])
            c[i]=3*lambda_*(y[i]-y[i-1])/(x[i]-x[i-1])+\
                 3*u*(y[i+1]-y[i])/(x[i+1]-x[i])
            A[i,i+1],A[i,i-1]=u,lambda_
        c[0]=3*(y[1]-y[0])/(x[1]-x[0])-(x[1]-x[0])*d2y[0]/2    #边界条件
        c[-1]=3*(y[-1]-y[-2])/(x[-1]-x[-2])-(x[-1]-x[-2])*d2y[-1]/2    #边界条件
        m=np.linalg.solve(A,c)
        self.__spline_poly__(t,x,y,m)     #构造三次样条插值多项式

    def __natural_spline__(self,t,x,y):
        """
        自然边界条件的三次样条插值,2阶导数值为0
        """
        d2y=np.array([0,0])
        self.__second_spline__(t,x,y,d2y)
    
    def __periodic_spline__(self,t,x,y):
        """
        periodic边界条件的三次样条插值
        """
        A=np.diag(2*np.ones(self.n-1))
        #边界特殊情况
        h0,h1,he=x[1]-x[0],x[2]-x[0],x[-1]-x[-2]
        A[0,1]=h0/(h0+h1)   #表示u_1
        A[0,-1]=1-A[0,1]    #表示lambda_1
        A[-1,0]=he/(he+h0)    #表示u_n
        A[-1,-2]=1-A[-1,0]    #表示lambda_n
        c=np.zeros(self.n-1)
        for i in range(1,self.n-1):
            u=(x[i]-x[i-1])/(x[i+1]-x[i-1])
            lambda_=(x[i+1]-x[i])/(x[i+1]-x[i-1])
            c[i-1]=3*lambda_*(y[i]-y[i-1])/(x[i]-x[i-1])+\
                 3*u*(y[i+1]-y[i])/(x[i+1]-x[i])
            if i < self.n-2:
                A[i,i+1],A[i,i-1]=u,lambda_
        c[-1]=3*(he*(y[1]-y[0])/h0+h0*(y[-1]-y[-2])/he)/(he+h0)  #边界条件
        m=np.zeros(self.n)
        m[1:]=np.linalg.solve(A,c)
        m[0]=m[-1]      #周期边界条件
        self.__spline_poly__(t,x,y,m)     #构造三次样条插值多项式

    def cal_interp(self,x0):
        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """
        from utils import piecewise_utils
        self.y0=piecewise_utils.cal_interp(self.polynomial,self.x,x0)
        return self.y0

    def plt_interp(self,x0=None,y0=None):
        """
        可视化插值图像和所求的插值点
        """
        from utils import piecewise_utils
        title=""
        if self.boundary_type == "complete":     #给定边界一阶导数相等
            title="Complete"
        elif self.boundary_type == "second":     #给定边界二阶导数相等
            title="Second"
        elif self.boundary_type == "natural":     #自然边界条件
            title="Natural"
        elif self.boundary_type == "periodic":     #周期边界条件
            title="Periodic"
        params=(self.polynomial,self.x,self.y,"Cubic Spline(%s)"% title,x0,y0)
        piecewise_utils.plt_interp(params)

if __name__ == "__main__":
    
    # 三组测试用数据
    # x=np.linspace(0,24,13,endpoint=True)
    # y=np.array([12,9,9,10,18,24,28,27,25,20,18,15,13])
    # x0=np.array([1,10.5,13,18.7,22.3])
    
    # x = np.linspace(0,2*np.pi,30)
    # y = 2*np.exp(-x)*np.sin(x)
    # dy = 2*np.exp(-x)*(np.cos(x)-np.sin(x))
    # d2y = 2*np.exp(-x)*(-2*np.cos(x)+np.sin(x))
    # x0=np.array([np.pi/2,2.158,3.58,4.784])

    x=np.linspace(0,2*np.pi,10,endpoint=True)
    y=np.sin(x)*np.cos(x)
    dy=np.cos(x)
    d2y=-np.sin(x)
    x0=np.array([np.pi/2,2.158,3.58,4.784])
    
    # 三次样条插值调用示例
    csi_Interp=CubicSplineInterpolation(x,y,dy=None,d2y=None,boundary_type='periodic')
    csi_Interp.fit_interp()
    print(csi_Interp.poly_coefficient)
    print(csi_Interp.polynomial)
    y0=csi_Interp.cal_interp(x0)
    print('所求插值点的值为：',y0)
    csi_Interp.plt_interp(x0,y0)