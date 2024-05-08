# Digital Waveform Generators

## Sinusoidal Generators

### 利用编程语言提供的`sin()`函数

基本上各种编程语言都会提供计算三角函数的基本的数学库，Python在math包里，C语言在<math.h>里。那么`sin()`是如何实现的呢？很可能是基于泰勒展开的。

### 利用z变换对

已知如下z变换对：

$$ \begin{flalign}
& \left[ \sin \omega_0 n \right] u\left[ n \right] \leftrightarrow \frac{\left[ \sin \omega_0 \right] z^{-1}}{1-[2\cos \omega_0]z^{-1} + z^{-2}} &
\end{flalign} $$

当输入信号$x[n]=\delta[n]$，输出信号$y[n]=\sin(\omega_0 n)u[n]$时，系统的传递函数为

$$ \begin{flalign}
& H(z) = \frac{Y(z)}{X(z)} = \frac{\left[ \sin \omega_0 \right] z^{-1}}{1-[2\cos \omega_0]z^{-1} + z^{-2}} &
\end{flalign} $$

由上式很容易写出系统的差分方程：

$$ \begin{flalign}
& y[n] = (\sin\omega_0) x[n-1] + 2(\cos\omega_0) y[n-1] - y[n-2] &
\end{flalign} $$

初始条件：

$$ \begin{flalign}
& y[0] = 0, & \\
& y[1] = (\sin\omega_0)x[0] &
\end{flalign} $$

### 利用数字谐振器

在江志宏的《学入浅出数字信号处理》一书中，讲到数字谐振器时，以生成DTMF信号为应用实例。读到此处时有两个问题没有搞懂：

1. 对于如下的系统传递函数，在产生正弦波时，如何确定的$r_p=1$呢？
2. 对于如下的系统传递函数，在产生正弦波时，如何确定的$G = \sin(\omega_0)$呢？

$$ \begin{flalign}
& H(z) 
    = \frac{G}{\left(1-r_p{\rm e}^{{\rm j}\omega_0}z^{-1} \right) \left(1-r_p{\rm e}^{-{\rm j}\omega_0}z^{-1} \right)} 
    = \frac{G}{1-2r_p \cos\left( \omega_0 \right)z^{-1} + r_p^2 z^{-2}} &
\end{flalign} $$

## Periodic Waveform Generators

已知某周期信号的周期为4，则

$$ \begin{flalign}
& h[n] = [b_0, b_1, b_2, b_3, b_0, b_1, b_2, b_3, ...] &
\end{flalign} $$

对其进行$z$变换

$$ \begin{flalign}
H[z] &= (b_0 + b_1 z^{-1} + b_2 z^{-2} + b_3 z^{-3})(1 + z^{-4} + z^{-8} + ...) \\
     &= \frac{b_0 + b_1 z^{-1} + b_2 z^{-2} + b_3 z^{-3}}{1-z^{-4}} &
\end{flalign} $$

按照上面的传递函数设计滤波器即可。

## Wavetable Generators

将一个周期内的信号写到表里，后面根据需求查表即可。

- 产生周期信号，像环形buffer那样不断查表即可
- 信号频率变为原来的两倍，隔一个采样点取一个
- 还可以进行幅值的调制
