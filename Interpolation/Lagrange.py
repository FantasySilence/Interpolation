import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class LagrangeInterpolation:
    """
    拉格朗日插值
    """

    def __init__(self, x, y):
        """
        拉格朗日插值必要的参数初始化
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
    
    def fit_interp(self):
        """
        生成拉格朗日插值多项式
        """
        t=sp.symbols('t')       #定义符号变量
        self.polynomial=0.0
        for i in range(self.n):
            # 针对每个数值点构造插值基函数
            basis_fun=self.y[i]  #插值基函数
            for j in range(i):
                basis_fun*=(t-self.x[j])/(self.x[i]-self.x[j])
            for j in range(i+1,self.n):
                basis_fun*=(t-self.x[j])/(self.x[i]-self.x[j])
            self.polynomial+=basis_fun
        
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
        params=(self.polynomial,self.x,self.y,'Lagrange',x0,y0)
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

    # 拉格朗日插值调用示例
    lag_Interp=LagrangeInterpolation(x=x,y=y)
    lag_Interp.fit_interp()
    print('拉格朗日插值多项式如下:')
    print(lag_Interp.polynomial)
    print('拉格朗日插值多项式系数向量和对应阶次:')
    print(lag_Interp.poly_coefficient)
    print(lag_Interp.coefficient_order)
    y1=lag_Interp.cal_interp(x0)
    print('所求插值点的值为：',y1)
    lag_Interp.plt_interp(x0)