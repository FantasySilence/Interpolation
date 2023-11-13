import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class NewtonDiffInterpolation:
    """
    牛顿差分插值
    实际中主要有，牛顿前插公式，牛顿后插公式
    """

    def __init__(self, x, y, diff_type='forward'):
        """
        牛顿差分插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """
        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.diff_type=diff_type   #差分方式,分为向前差分(forward)和向后差分(backward)
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从高到低
        self.coefficient_order=None   #对应多项式系数的阶次
        self.y0=None   #所求插值点的值，单个值或者向量
        self.diff_mat=None     #储存离散数据点的差分
        self.__x__=None     #牛顿差分起始的x坐标值
        self.h=None    #计算牛顿差分插值时，计算x坐标的等距步长

    def __check_equidistance__(self,x):
        """
        检查数据点x是否是等距的
        x:已知数据点
        """
        xx=np.linspace(min(x),max(x),len(x),endpoint=True)
        if (x==xx).all() or (x==xx[::-1]).all():
            self.h = x[1]-x[0]      #等距步长
        else:
            raise ValueError("数据点不是等距的,不适合使用牛顿差分插值！")
        
    def __diff_matrix__(self):
        """
        计算牛顿差分矩阵
        """
        self.__check_equidistance__(self.x)     #判断是否等距
        self.diff_mat=np.zeros((self.n,self.n))
        self.diff_mat[:,0]=self.y   #差分矩阵的第一列存储y值，即0阶差分
        if self.diff_type=='forward':
            for j in range(1,self.n):
                for i in range(self.n-j):
                    self.diff_mat[i,j]=self.diff_mat[i+1,j-1]-self.diff_mat[i,j-1]
        elif self.diff_type=='backward':
            for j in range(1,self.n):
                for i in range(j,self.n):
                    self.diff_mat[i,j]=self.diff_mat[i,j-1]-self.diff_mat[i-1,j-1]
        else:
            raise ValueError("diff_type参数错误,差分形式只适合前向forward和后向backward！")
    
    def fit_interp(self):
        """
        构造牛顿差分插值多项式
        """
        t=sp.Symbol('t')
        term,factorial=t,1      #差分项 # type: ignore
        self.__diff_matrix__()      #计算差分矩阵
        if self.diff_type=='forward':
            self.__x__=self.x[0]    #第一个值x0
            df=self.diff_mat[0,:]   #前向差分只用第一行的值   # type: ignore
            self.polynomial=df[0]   #初始化y0的值
            for i in range(1,self.n):
                self.polynomial += df[i]*term/factorial
                term *= (t-i)       # type: ignore
                factorial *= (i+1)  
        elif self.diff_type=='backward':
            self.__x__=self.x[-1]       #最后一个值xn
            db=self.diff_mat[-1,:]   #后向差分只用最后一行的值   # type: ignore
            self.polynomial=db[0]   #初始化yn的值
            for i in range(1,self.n):
                self.polynomial += db[i]*term/factorial
                term *= (t+i)       # type: ignore
                factorial *= (i+1)
        
        # 插值多项式特征
        self.polynomial=sp.expand(self.polynomial)  #多项式展开
        polynomial=sp.Poly(self.polynomial,t)       #构造多项式对象
        self.poly_coefficient=polynomial.coeffs()   #获取多项式的系数
        self.coefficient_order=polynomial.monoms()  #多项式系数对应的阶次
    
    def cal_interp(self,x0):
        """
        计算给定插值点的数值
        x0:所求插值的x坐标值
        """
        from utils import Interp_utils
        t0=(x0-self.__x__)/self.h
        self.y0=Interp_utils.cal_interp(self.polynomial,t0)
        return self.y0
    
    def plt_interp(self,x0=None,y0=None):
        """
        可视化插值图像和插值点
        """
        plt.figure(figsize=(8,6))
        plt.plot(self.x,self.y,'ro',label='Interp points')
        xi=np.linspace(min(self.x),max(self.x),100)
        yi=self.cal_interp(xi)
        plt.plot(xi,yi,'b--',label='Interpolation')
        if x0 is not None and y0 is not None:
            plt.plot(x0,y0,'g*',label='Cal points')
        plt.legend()
        plt.xlabel('x',fontdict={'fontsize':12})
        plt.ylabel('y',fontdict={'fontsize':12})
        if self.diff_type == 'forward':
            plt.title('NewtonForward Interpolation',fontdict={'fontsize':14})
        elif self.diff_type == 'backward':
            plt.title('NewtonBackward Interpolation',fontdict={'fontsize':14})
        plt.grid(linestyle=':')
        plt.show()

if __name__ == "__main__":
    
    # 三组测试用数据
    x=np.linspace(0,24,13,endpoint=True)
    y=np.array([12,9,9,10,18,24,28,27,25,20,18,15,13])
    x0=np.array([1,10.5,13,18.7,22.3])
    
    # x = np.linspace(0,2*np.pi,30)
    # y = 2*np.exp(-x)*np.sin(x)
    # dy = 2*np.exp(-x)*(np.cos(x)-np.sin(x))
    # d2y = 2*np.exp(-x)*(-2*np.cos(x)+np.sin(x))
    # x0=np.array([np.pi/2,2.158,3.58,4.784])

    # x=np.linspace(0,2*np.pi,10,endpoint=True)
    # y=np.sin(x)*np.cos(x)
    # dy=np.cos(x)
    # d2y=-np.sin(x)
    # x0=np.array([np.pi/2,2.158,3.58,4.784])
    
    # 牛顿差分插值调用示例
    Ndi_Interp=NewtonDiffInterpolation(x=x,y=y,diff_type='backward')
    Ndi_Interp.__diff_matrix__()
    Ndi_Interp.fit_interp()
    print('牛顿差分插值多项式如下:')
    print(Ndi_Interp.polynomial)
    print('牛顿差分插值多项式系数向量和对应阶次:')
    print(Ndi_Interp.poly_coefficient)
    print(Ndi_Interp.coefficient_order)
    y3=Ndi_Interp.cal_interp(x0)
    print('所求插值点的值为：',y3)
    Ndi_Interp.plt_interp(x0,y3)