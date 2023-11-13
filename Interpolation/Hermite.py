import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class HermiteInterpolation:
    """
    求解带1阶导数值的埃尔米特插值多项式
    """

    def __init__(self, x, y,dy):
        """
        埃尔米特插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """
        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        self.dy = np.asarray(dy,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y) and len(self.y)==len(self.dy):
            self.n=len(self.x)
        else:
            raise ValueError("x,y,dy坐标长度不匹配")
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从高到低
        self.coefficient_order=None   #对应多项式系数的阶次
        self.y0=None   #所求插值点的值，单个值或者向量
    
    def fit_interp(self):
        """
        生成埃尔米特插值多项式的核心算法
        """
        t=sp.Symbol('t')    #定义符号变量
        self.polynomial=0.0     #埃尔米特插值多项式初始化
        for i in range(self.n):
            hi,ai=1.0,0.0   #插值多项式的辅助函数构造
            for j in range(self.n):
                if i!=j:
                    hi *= ((t-self.x[j])/(self.x[i]-self.x[j]))**2
                    ai += 1/(self.x[i]-self.x[j])
            self.polynomial += hi*((self.x[i]-t)*(2*ai*self.y[i]-self.dy[i])+self.y[i])
        
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
        self.y0=Interp_utils.cal_interp(self.polynomial,x0)
        return self.y0
    
    def plt_interp(self,x0=None,y0=None):
        """
        可视化插值图像和插值点
        """
        from utils import Interp_utils
        params=(self.polynomial,self.x,self.y,'Hermite',x0,y0)
        Interp_utils.plt_interp(params)



class Piecewise2CubicHermiteInterpolation:
    """
    两点三次埃尔米特插值
    """
    def __init__(self, x, y,dy):
        """
        两点三次埃尔米特必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        dy:一阶导数的值
        """
        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        self.dy = np.asarray(dy,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y) and len(self.y)==len(self.dy):
            self.n=len(self.x)
        else:
            raise ValueError("x,y,dy坐标长度不匹配")
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从3到低
        self.y0=None   #所求插值点的值，单个值或者向量
    
    def fit_interp(self):
        """
        生成两点三次埃尔米特插值多项式
        """
        t=sp.Symbol('t')    #定义符号变量
        self.polynomial=dict()     #三次多项式函数
        self.poly_coefficient=np.zeros((self.n-1,4))
        for i in range(self.n-1):
            hi=self.x[i+1]-self.x[i]    #每个小区间的长度
            #两点三次埃尔米特插值的函数公式
            poly23_i=self.y[i]*(1+2*(t-self.x[i])/hi)*((t-self.x[i+1])/hi)**2+\
                     self.y[i+1]*(1-2*(t-self.x[i+1])/hi)*((t-self.x[i])/hi)**2+\
                     self.dy[i]*(t-self.x[i])*((t-self.x[i+1])/hi)**2+\
                     self.dy[i+1]*(t-self.x[i+1])*((t-self.x[i])/hi)**2
            self.polynomial[i]=sp.simplify(poly23_i)
            poly_obj=sp.Poly(poly23_i,t)      #构造多项式对象
            #某项系数可能为0，为防止存储错误，分别对应各阶次存储
            mons=poly_obj.monoms()    #多项式系数对应的阶次
            for j in range(len(mons)):
                self.poly_coefficient[i,mons[j][0]]=poly_obj.coeffs()[j]  #获取多项式的系数
    
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
        params=(self.polynomial,self.x,self.y,"2 Point Cubic Hermite",x0,y0)
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
    
    # 埃尔米特插值(带1阶导数值)调用示例
    her_Interp=HermiteInterpolation(x=x,y=y,dy=dy)
    her_Interp.fit_interp()
    print('埃尔米特插值多项式如下:')
    print(her_Interp.polynomial)
    print('埃尔米特插值多项式系数向量和对应阶次:')
    print(her_Interp.poly_coefficient)
    print(her_Interp.coefficient_order)
    y0=her_Interp.cal_interp(x0)
    print('所求插值点的值为：',y0)
    her_Interp.plt_interp(x0,y0)

    # 两点三次埃尔米特插值调用示例
    her2_Interp=Piecewise2CubicHermiteInterpolation(x,y,dy)
    her2_Interp.fit_interp()
    print(her2_Interp.poly_coefficient)
    y0=her2_Interp.cal_interp(x0)
    print('所求插值点的值为：',y0)
    her2_Interp.plt_interp(x0,y0)