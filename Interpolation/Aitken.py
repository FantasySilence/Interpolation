import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class AitkenStepwiseInterpolation:
    """
    艾特肯逐步插值
    不要求精度，逐步递推到最后一个多项式
    """

    def __init__(self, x, y):
        """
        艾特肯逐步插值必要的参数初始化
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
        self.aitken_mat=None
    
    def fit_interp(self):
        """
        生成艾特肯逐步插值多项式
        """
        t=sp.symbols('t')       #定义符号变量
        self.aitken_mat=sp.zeros(self.n,self.n+1)
        self.aitken_mat[:,0],self.aitken_mat[:,1]=self.x,self.y
        poly_next=[t for _ in range(self.n)]  #储存下一列递推多项式
        poly_before=np.copy(self.y) #储存上一列递推多项式
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                poly_next[j]=(poly_before[j]*(t-self.x[i])-poly_before[i]*(t-self.x[j]))\
                    /(self.x[j]-self.x[i])
            poly_before=poly_next
            self.aitken_mat[i+1:,i+2]=poly_next[i+1:]
        self.polynomial=poly_next[-1]    #艾特肯递推完成后的最后一个多项式，即最终
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
        params=(self.polynomial,self.x,self.y,'Aitken Stepwise',x0,y0)
        Interp_utils.plt_interp(params)



class AitkenInterpolationWithEpsilon:
    """
    艾特肯逐步插值
    带要求精度,未必逐步递推到最后一个多项式，只要达到精度要求即可
    """

    def __init__(self, x, y, esp=1e-3):
        """
        艾特肯逐步插值必要的参数初始化
        x:已知数据的x坐标
        y:已知数据的y坐标
        """
        import numpy as np
        self.x = np.asarray(x,dtype=np.float64)
        self.y = np.asarray(y,dtype=np.float64)
        if len(self.x)>1 and len(self.x)==len(self.y):
            self.n=len(self.x)
        else:
            raise ValueError("x,y坐标长度不匹配")
        self.esp=esp   #插值点的精度要求
        self.y0=None   #所求插值点的值，单个值或者向量
        self.recurrence_num=None  #储存每个插值点的递推次数
    
    def fit_interp(self,x0):
        """
        根据给定的插值点x0，并根据精度要求，逐步递推
        """
        x0=np.asarray(x0,dtype=np.float64)
        y_0=np.zeros(len(x0))    #用于储存对应x0的插值
        self.recurrence_num=[]
        for k in range(len(x0)):    #针对每个插值点逐个计算
            val_next=np.zeros(self.n)  #储存下一列递推多项式的值
            val_before=np.copy(self.y) #储存上一列递推多项式的值
            tol,i=1,0   #初始精度要求
            for i in range(self.n-1):
                for j in range(i+1,self.n):
                    val_next[j]=(val_before[j]*(x0[k]-self.x[i])-val_before[i]*(x0[k]-self.x[j]))\
                        /(self.x[j]-self.x[i])
                tol=np.abs(val_before[i+1]-val_next[i+1])
                val_before[i+1:]=val_next[i+1:]
                if tol<=self.esp:   #满足精度要求之后退出循环，不再进行递推
                    break
            y_0[k]=val_next[i+1]    #满足精度要求的插值存储
            self.recurrence_num.append(i+1)  #储存每个插值点的递推次数
        self.y0=y_0     #计算完毕后，赋值给类属性变量以供调用
        return y_0
    
    def plt_interp(self,x0,y0):
        """
        可视化插值图像和插值点
        """
        plt.figure(figsize=(8,6))
        plt.plot(self.x,self.y,'ro',label='Interp points')
        xi=np.linspace(min(self.x),max(self.x),100)
        yi=self.fit_interp(xi)
        plt.plot(xi,yi,'b--',label='Interpolation')
        if x0 is not None and y0 is not None:
            plt.plot(x0,y0,'g*',label='Cal points')
        plt.legend()
        plt.xlabel('x',fontdict={'fontsize':12})
        plt.ylabel('y',fontdict={'fontsize':12})
        avg_recurrence=np.round(np.mean(self.recurrence_num),2) # type: ignore
        plt.title('Aitken Interpolation with Epsilon=%.1e,Avg Recurrence=%.1f'
                  %(self.esp,avg_recurrence),fontdict={'fontsize':13})
        plt.grid(linestyle=':')
        plt.show()

if __name__ == "__main__":
    pass
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

    # 艾特肯逐步插值调用示例
    asi_Interp=AitkenStepwiseInterpolation(x=x,y=y)
    asi_Interp.fit_interp()
    print('艾特肯逐步插值多项式如下:')
    print(asi_Interp.polynomial)
    print('艾特肯逐步插值多项式系数向量和对应阶次:')
    print(asi_Interp.poly_coefficient)
    print(asi_Interp.coefficient_order)
    y2=asi_Interp.cal_interp(x0)
    print('所求插值点的值为：',y2)
    asi_Interp.plt_interp(x0)

    # 带精度的艾特肯逐步插值调用示例
    asi_Interp=AitkenInterpolationWithEpsilon(x=x,y=y)
    y0=asi_Interp.fit_interp(x0)
    print('所求插值点的值为：',y0)
    print('每个插值点递推次数为：',asi_Interp.recurrence_num)
    asi_Interp.plt_interp(x0,y0)
