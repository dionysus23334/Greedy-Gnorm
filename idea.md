# 曲线积分证明解决下溢出方案保序

$$
F(a_1,a_2,...,a_n)  = \sum^{n}_{i=1}a_ilog(a_i)
$$
在$R^{n+}$连续
$$
\sum^{n}_{i=1}x_i=\sum^{n}_{i=1}y_i=1
$$
若
 $P_1=(x_1,x_2,...,x_n)$
$P_2=(y_1,y_2,...,y_n)$
$$
F(P_2)-F(P_1)=\int_{P_1}^{P_2}\bigtriangledown F ·\vec{l}\ dl>0
$$
则
$P_1'=(x_1+\varepsilon,x_2+\varepsilon,...,x_n+ \varepsilon )$
$P_2'=(y_1+\varepsilon,y_2+\varepsilon,...,y_n+\varepsilon)$

$$
F(P_2')-F(P_1')=\int_{P_1'}^{P_2'}\bigtriangledown F' ·\vec{l}\ dl
$$
这里l没有变化，因为被积的线段不变

$$
\vec l=(l_1,l_2,...,l_n)=\frac{\vec{P_1P_2}}{||\vec{P_1P_2}||}=\frac{\vec{P'_1P'_2}}{||\vec{P'_1P'_2}||}
$$
$\sum^{n}_{i=1}l_ilog(a_i)=\bigtriangledown F·\vec l>0$ 

因为函数连续，所以当$\varepsilon$足够小，则落在小邻域内，梯度符号保持不变
$\bigtriangledown F'·\vec l=\sum^{n}_{i=1}l_ilog(a_i+\varepsilon) \in N_r(\bigtriangledown F·\vec l)$
$$
F(P_2')>F(P_1')
$$


完成了注意力熵的修正工作，即下溢出问题的纠正，还有用曲线积分证明了e足够小时纠正后的注意力熵的保序性。

$n大于等于2$
$A(a_1,a_2,...,a_n)  = \sum^{n}_{i=1}a_ilog(a_i)$
$B(a_1,a_2,...,a_n)  = \sum^{n}_{i=1}a_ilog(a_i+\varepsilon)$
$C(a_1,a_2,...,a_n)  = \sum^{n}_{i=1}(a_i+\varepsilon)log(a_i+\varepsilon)$
这里$\varepsilon$为小量（一定小于0.5）
$$
C-A=C-B+B-A
$$
$显然C-B<0;B-A>0$
这里$C-B<0$由
$$
(\prod_i (a_i+\varepsilon))^{\frac{1}{n}}<\frac{n\varepsilon+1}{n}<1
$$
得出。
因此C离A更近

这里的单个rectified表示B的AE计算方式，两个rectified表示C的AE计算方式

可以考虑用交叉熵来剪头。

交叉熵的定义需要好好想想。

# 梯度矩阵的计算，贪婪算法剪头的基础

用Frobenius norm来定义一个梯度矩阵的范数。然后做如下的计算，来衡量一个头的梯度分数大小，每剪一个头都要算一次，得到不同k下的score，当然这里要在不同的数据下取均值。

$$S(k)=G_{Q(k)}\odot G^T_{K(k)}\odot G_{V(k)}$$
$$
G_{Q(k)}=E(G_{q(k)}) 
$$
$$
G_{q(k)}=
\begin{pmatrix}
  ||G^{(11)}_{q(k)}|| & ||G^{(12)}_{q(k)}|| & \cdots & ||G^{(1H)}_{q(k)}|| \\
  ||G^{(21)}_{q(k)}|| & ||G^{(22)}_{q(k)}|| & \cdots & ||G^{(2H)}_{q(k)}|| \\
  \vdots & \vdots & \ddots & \vdots \\
  ||G^{(L1)}_{q(k)}|| & ||G^{(L2)}_{q(k)}|| & \cdots & ||G^{(LH)}_{q(k)}||
\end{pmatrix}
\in \mathbb{R}^{L \times H}
$$


$$

||G^{(ij)}_{q(k)}||=\sqrt{\sum_m(\frac{\partial ||F(X)||}{\partial q^{(ij)}_m})^2}

$$

$$
G_{Q(k)}=E(G_{q(k)})=\begin{pmatrix}
  \frac{1}{||X||}\sum_{x\in X}\sqrt{\sum_m(\frac{\partial ||F(x)||}{\partial q^{(11)}_m})^2}
 &  \cdots &  \\
  \vdots  & \ddots &\vdots \\
   &\cdots & \frac{1}{||X||}\sum_{x\in X}\sqrt{\sum_m(\frac{\partial ||F(x)||}{\partial q^{(LH)}_m})^2}
\end{pmatrix}
\in \mathbb{R}^{L \times H}

$$
# 基于梯度矩阵范数的贪婪裁剪方法

现在由于思路出现了分岔口，可以开始做两个研究，分别实验对比熵剪法和最大梯度剪法，还有最大梯度剪法和范数贪婪法


# U裁剪

验证 $$\frac{\frac{\partial F}{\partial W_{K_j}}}{\frac{\partial u_{ij}}{\partial W_{K_j}}} = \frac{\frac{\partial F}{\partial W_{Q_i}}}{\frac{\partial u_{ij}}{\partial W_{Q_i}}}=\frac{\partial F}{\partial u_{ij}}$$
取出grad计算后发现并不等，所以这里选择等式

$$\frac{\partial F}{\partial u_{ij}}=\frac{\frac{\partial F}{\partial W_{K_j}}+\frac{\partial F}{\partial W_{Q_i}}}{\frac{\partial u_{ij}}{\partial W_{K_j}}+\frac{\partial u_{ij}}{\partial W_{Q_i}}}$$

$W_{Q_i}=[q_{i1},q_{i2},...,q_{id_k}]^T$
$W_{K_j}=[k_{j1},k_{j2},...,k_{jd_k}]^T$
$$ W_Q,W_K \in \mathbb{R}^{d_{model} \times d_k}
$$
$$
Att=Softmax(\frac{XW_QW_K^TX^T}{\sqrt d_k})XW_V
$$
$$U=W_QW_K^T=[u_{ij}]_{d_{model} \times d_{model}}$$
$$u_{ij}=\sum^{d_k}_{m=1}q_{im}k_{jm}$$
实现不了，运行时间可能会非常长


下次科研可以着重放在：验证了梯度是实时变化的，除此之外目前的想法就是改善优化梯度裁剪方式，使用贪婪算法剪掉最小的梯度的头。
这次可以提一下实时变化。