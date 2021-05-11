import { Matrix } from "../src"

export class Optimize {
  /**
   * 交叉墒代价函数
   * 用于最后一层激活函数是sigmoid
   * 输出值域必须是 {0, 1}
   * - J = 1 / m * ∑m Cost
   * - y = 1 ? Cost = - Math.log(H(X[i]))
   * - y = 0 ? Cost = -Math.log(1 - H(X[i]))
   */
  crossCost(hy: Matrix, ys: Matrix) {
    let m = ys.shape[0]
    let t = hy.atomicOperation((h, i, j) => {
      let y = ys.get(i, j)
      return y === 1 ? -Math.log(h) : -Math.log(1 - h)
    }).columnSum()
    let tmp = t.getRow(0).map(v => v / m)
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * Momentum
   * 动量的梯度下降法
   * - z = 0.9
   * - v[t] = z * v[t-1] + α * (∂J / ∂w)
   * - w = w - v[t]
   */
  momentum(xs: Matrix[], ys: Matrix) { }
  /**
   * AdaGrad
   * 学习衰减率法，训练初期波动较大，
   * 后期学习率可能趋近于0，导致不更新梯度
   * 将以前的梯度求平方和存储起来
   * - h[t] = h[t-1] + (∂J / ∂w) ** 2
   * - w = w - (α / (Math.sqrt(h[t]) + 1e-07)) * (∂J / ∂w)
   */
  adaGrad(xs: Matrix[], ys: Matrix) { }
  /**
   * adaDelta AdaGrad改进版
   * 解决后期学习率趋近于0的情况
   * 增加一个衰减系数z，一般z = 0.999
   * - h[t] = z * h[t-1] + (1- z)(∂J / ∂w) ** 2
   * - w = w - (α / (Math.sqrt(h[t]) + 1e-07)) * (∂J / ∂w)
   */
  adaDelta(xs: Matrix[], ys: Matrix) { }
}