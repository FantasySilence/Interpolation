import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class NewtonInterpolation:

    """
    牛顿插值
    """

    def __init__(self, x, y):
        """
        牛顿插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """
        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.polynomial=None   #最终的插值多项式的符号表示
        self.poly_coefficient=None   #最终的插值多项式的系数，幂次从高到低
        self.coefficient_order=None   #对应多项式系数的阶次
        self.y0=None   #所求插值点的值，单个值或者向量
        self.diff_quot=None     #储存离散数据点的差商
    
    def __diff_quotient__(self):
        """
        计算牛顿均差（差商）
        """
        diff_quot=np.zeros((self.n,self.n))
        diff_quot[:,0]=self.y   #第一列存储y值
        for j in range(1,self.n):    #按列计算
            for i in range(j,self.n):   #行，初始值为差商表的对角线值
                diff_quot[i,j]=(diff_quot[i,j-1]-diff_quot[i-1,j-1])/(self.x[i]-self.x[i-j])
        self.diff_quot=pd.DataFrame(diff_quot)
        return diff_quot
    
    def fit_interp(self):
        """
        生成牛顿插值多项式
        """
        t=sp.symbols('t')       #定义符号变量
        diff_quot=self.__diff_quotient__()  #计算差商表
        d_q=np.diag(diff_quot)      #构造牛顿插值时只需要对角线元素
        self.polynomial=d_q[0]
        term_poly=t-self.x[0]
        for i in range(1,self.n):
            self.polynomial+=d_q[i]*term_poly
            term_poly*=(t-self.x[i])
        
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
        params=(self.polynomial,self.x,self.y,'Newton',x0,y0)
        Interp_utils.plt_interp(params)

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
    
    # 牛顿差商插值调用示例
    new_Interp=NewtonInterpolation(x=x,y=y)
    new_Interp.fit_interp()
    print('牛顿插值多项式如下:')
    print(new_Interp.polynomial)
    print('牛顿插值多项式系数向量和对应阶次:')
    print(new_Interp.poly_coefficient)
    print(new_Interp.coefficient_order)
    y0=new_Interp.cal_interp(x0)
    print('所求插值点的值为：',y0)
    new_Interp.plt_interp(x0)