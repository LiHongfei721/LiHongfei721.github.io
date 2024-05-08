# ChirpZ算法

DFT可以看作是对z变换结果$X(z)$在单位圆上的等间隔采样。在信号处理中，除了计算整个单位圆上的$X(z)$之外，也常常需要计算$X(z)$在$z$平面上其他曲线上值，这时的快速计算都要用到ChirpZ算法。用ChirpZ算法可以更好地分析感兴趣带宽内的频谱。

## 算法原理


