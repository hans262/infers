import { ActivationFunction } from "./BPNet";
import { Matrix } from "./matrix";

export class Layer {
  w: Matrix
  b: Matrix
  hy?: Matrix
  constructor(
    public index: number, //当前层的索引 0 ~ n
    public unit: number,  //神经元个数
    public lastUnit: number, //上一层的神经元个数
    public lastLayer?: Layer, //上一层的对象，第一个隐藏层没有
    public af?: ActivationFunction
  ) {
    //初始化权重偏值
    this.w = Matrix.generate(unit, this.lastUnit)
    this.b = Matrix.generate(1, unit)
  }

  calchy(xs?: Matrix) {
    //用上一层的结果 * 当前层的权值 + 偏值
    if (this.index === 1) {
      this.hy = xs
    } else {
      this.hy = this.lastLayer!.hy!.multiply(this.w.T).atomicOperation((item, _, j) =>
        this.afn(item + this.b.get(0, j))
      )
    }
  }

  /**
   * Get activation function for current layer
   */
  afn(x: number) {
    switch (this.af) {
      case 'Sigmoid':
        return 1 / (1 + Math.exp(-x))
      case 'Relu':
        return x >= 0 ? x : 0
      case 'Tanh':
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
      default:
        return x
    }
  }

  /**
   * Gets the active function derivative of the current layer
   */
  afd(x: number) {
    switch (this.af) {
      case 'Sigmoid':
        return x * (1 - x)
      case 'Relu':
        return x >= 0 ? 1 : 0
      case 'Tanh':
        return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2
      default:
        return 1
    }
  }
}