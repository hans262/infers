export class Matrix {
  shape: [number, number]
  private self: number[][]

  constructor(data: number[][]) {
    let t = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length)
    if (t) throw new Error('矩阵列不正确')
    this.shape = [data.length, data[0].length]
    this.self = data
  }

  /**
   * 矩阵mxn每列求和
   * @returns Matrix 1 x n
   */
  columnSum() {
    let n = []
    for (let i = 0; i < this.shape[1]; i++) {
      n.push(this.getCol(i).reduce((p, c) => p + c))
    }
    return new Matrix([n])
  }
  
  /**
   * 返回拷贝后的二维数组
   */
  dataSync() {
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        m.push(this.get(i, j))
      }
      n.push(m)
    }
    return n
  }

  /**
   * 对比矩阵的形状
   * @param b 
   */
  equalsShape(b: Matrix) {
    return this.shape[0] === b.shape[0] && this.shape[1] === b.shape[1]
  }

  /**
   * 对比两个矩阵
   * @param b 
   */
  equals(b: Matrix) {
    if (!this.equalsShape(b)) {
      return false
    }
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        if (this.get(i, j) !== b.get(i, j)) return false
      }
    }
    return true
  }

  /**
   * 生成矩阵
   * @param row 
   * @param col 
   * @param f 默认为-0.5 ~ 0.5随机值
   */
  static generate(row: number, col: number, f?: number) {
    let n = []
    for (let i = 0; i < row; i++) {
      let m = []
      for (let j = 0; j < col; j++) {
        m.push(f ? f : 0.5 - Math.random())
      }
      n.push(m)
    }
    return new Matrix(n)
  }

  /**
   * 更新原矩阵
   * @param row 
   * @param col 
   * @param val 
   */
  update(row: number, col: number, val: number) {
    this.self[row][col] = val
  }

  /**
   * 向矩阵左右追加一列
   * @param n 
   * @param position 
   */
  expand(n: number, position: 'L' | 'R') {
    let m = []
    for (let i = 0; i < this.shape[0]; i++) {
      if (position === 'L') {
        m.push([n, ...this.getRow(i)])
      } else {
        m.push([...this.getRow(i), n])
      }
    }
    return new Matrix(m)
  }

  /**
   * 根据索引获取元素
   * @param i 
   * @param j 
   */
  get(i: number, j: number) {
    return this.self[i][j]
  }

  /**
   * 根据索引获取行
   * @param i 
   */
  getRow(i: number) {
    return [...this.self[i]]
  }

  /**
   * 根据索引获取列
   * @param k 
   */
  getCol(k: number) {
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        if (j === k) {
          n.push(this.get(i, j))
        }
      }
    }
    return n
  }

  /**
   * 行列式计算
   */
  det() {
    if (this.shape[0] !== this.shape[1]) {
      throw new Error('只有方阵才能计算行列式')
    }
    if (this.shape[0] === 2 && this.shape[1] === 2) {
      return this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0)
    } else {
      let m = 0
      //默认按照第一行的余因子展开计算
      for (let i = 0; i < this.shape[1]; i++) {
        if (this.get(0, i) !== 0) {
          m += this.get(0, i) * ((-1) ** (i + 2)) * this.cominor(0, i).det()
        }
      }
      return m
    }
  }

  /**
   * 筛选余子式
   * @param rowi 行
   * @param coli 列
   */
  cominor(rowi: number, coli: number) {
    if (this.shape[0] < 2 || this.shape[1] < 2) {
      throw new Error('求余子式行和列必须大于2才有意义')
    }
    let n = this.dataSync().map((v) => {
      v = v.filter((_, j) => j !== coli)
      return v
    }).filter((_, i) => i !== rowi)
    return new Matrix(n)
  }

  /**
   * 矩阵元素的原子操作
   * @param callback 
   */
  atomicOperation(callback: (item: number, row: number, col: number) => number) {
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        m.push(callback(this.get(i, j), i, j))
      }
      n.push(m)
    }
    return new Matrix(n)
  }

  /**
   * 同位操作
   * @param b 
   */
  coLocationOperation(b: Matrix, oper: 'add' | 'sub') {
    if (!this.equalsShape(b)) {
      throw new Error('必须满足两个矩阵是同形矩阵')
    }
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        let c = oper === 'add' ? this.get(i, j) + b.get(i, j) : this.get(i, j) - b.get(i, j)
        m.push(c)
      }
      n.push(m)
    }
    return new Matrix(n)
  }

  /**
   * 减法
   * @param b 
   */
  subtraction(b: Matrix) {
    return this.coLocationOperation(b, 'sub')
  }

  /**
   * 加法
   * @param b 
   */
  addition(b: Matrix) {
    return this.coLocationOperation(b, 'add')
  }

  /**
   * 数乘
   * @param b 
   */
  numberMultiply(b: number) {
    return this.atomicOperation(item => item * b)
  }

  /**
   * 乘法
   * @param b 
   */
  multiply(b: Matrix) {
    if (this.shape[1] !== b.shape[0]) {
      throw new Error('当矩阵A的列数等于矩阵B的行数，A与B才可以相乘')
    }
    let row = this.shape[0]
    let col = b.shape[1]
    let bt = b.T
    let n = []
    for (let i = 0; i < row; i++) {
      let m = []
      for (let k = 0; k < col; k++) {
        let tm = this.getRow(i).reduce((p, c, j) => {
          return p + c * bt.get(k, j)
        }, 0)
        m.push(tm)
      }
      n.push(m)
    }
    return new Matrix(n)
  }

  /**
   * 转置
   */
  get T() {
    let a = []
    for (let i = 0; i < this.shape[1]; i++) {
      let n = []
      for (let j = 0; j < this.shape[0]; j++) {
        n.push(this.get(j, i))
      }
      a.push(n)
    }
    return new Matrix(a)
  }

  /**
   * 归一化  
   * - X = X - average / range
   * @returns [归一化矩阵，缩放比矩阵]
   */
  normalization() {
    let t = this.T
    let n = []
    for (let i = 0; i < t.shape[0]; i++) {
      const max = Math.max(...t.getRow(i))
      const min = Math.min(...t.getRow(i))
      const range = max - min
      const average = min + (range / 2)
      n.push([average, range])
      for (let j = 0; j < t.shape[1]; j++) {
        let s = range === 0 ? 0 : (t.get(i, j) - average) / range
        t.update(i, j, s)
      }
    }
    return [t.T, new Matrix(n).T]
  }

  /**
   * 格式化输出矩阵
   */
  print() {
    console.log(`Matrix ${this.shape[0]}x${this.shape[1]} [`)
    for (let i = 0; i < this.shape[0]; i++) {
      let line = ' '
      for (let j = 0; j < this.shape[1]; j++) {
        line += this.get(i, j) + ', '
      }
      console.log(line)
    }
    console.log(']')
  }
}