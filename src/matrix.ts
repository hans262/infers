export type MatrixShape = [number, number]

export class Matrix {
  shape = [1, 1]
  private self = [[0]]
  constructor(data: number[][]) {
    let m = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length)
    if (m) throw new Error('矩阵列不正确')
    this.shape[0] = data.length
    this.shape[1] = data[0].length
    this.self = data
  }
  /**
   * 生成矩阵
   * @param row 
   * @param col 
   * @param f 
   * @returns 
   */
  static generate(row: number, col: number, f: number) {
    let n = []
    for (let i = 0; i < row; i++) {
      let m = []
      for (let j = 0; j < col; j++) {
        m.push(f)
      }
      n.push(m)
    }
    return new Matrix(n)
  }
  /**
   * 根据位置上的值
   * @param row 
   * @param col 
   * @param val 
   */
  update(row: number, col: number, val: number) {
    this.self[row][col] = val
  }

  /**
   * 向矩阵左右添加一列
   * @param n 
   * @param position 
   * @returns 
   */
  expansion(n: number, position: 'L' | 'R') {
    let m = []
    for (let i = 0; i < this.shape[0]; i++) {
      if (position === 'L') {
        m.push([n, ...this.getLine(i)])
      } else {
        m.push([...this.getLine(i), n])
      }
    }
    return new Matrix(m)
  }

  /**
   * 根据索引获取位置元素
   * @param i 
   * @param j 
   * @returns 
   */
  get(i: number, j: number) {
    return this.self[i][j]
  }
  /**
   * 根据索引获取行
   * @param i 
   * @returns 
   */
  getLine(i: number) {
    return this.self[i]
  }
  /**
   * 行列式
   * @returns 
   */
  det() {
    if (this.shape[0] !== this.shape[1]) {
      throw new Error('只有方阵才能计算行列式')
    }
    if (this.shape[0] === 2 && this.shape[1] === 2) {
      return this.self[0][0] * this.self[1][1] - this.self[0][1] * this.self[1][0]
    } else {
      let m = 0
      //按照第一行的余因子展开来计算
      for (let i = 0; i < this.shape[1]; i++) {
        if (this.get(0, i) !== 0) {
          m += this.get(0, i) * (-1) ** (i + 2) * this.cominor(1, i + 1).det()
        }
      }
      return m
    }
  }

  /**
   * 余子式
   * @param rowi 行
   * @param coli 列
   * @returns 
   */
  cominor(rowi: number, coli: number) {
    if (this.shape[0] < 2 || this.shape[1] < 2) {
      throw new Error('求余子式行和列必须大于2才有意义')
    }
    let n = this.self.map((v) => {
      v = v.filter((_, j) => j !== coli - 1)
      return v
    }).filter((_, i) => i !== rowi - 1)
    return new Matrix(n)
  }
  /**
   * 同位操作
   * @param b 
   */
  coLocationOperation(b: Matrix, oper: 'add' | 'sub') {
    if (this.shape[0] !== b.shape[0] || this.shape[1] !== b.shape[1]) {
      throw new Error('必须满足两个矩阵是同形矩阵')
    }
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        let c = oper === 'add' ? this.self[i][j] + b.self[i][j] : this.self[i][j] - b.self[i][j]
        m.push(c)
      }
      n.push(m)
    }
    return new Matrix(n)
  }
  /**
   * 减法
   * @param b 
   * @returns 
   */
  subtraction(b: Matrix) {
    return this.coLocationOperation(b, 'sub')
  }
  /**
   * 加法
   * @param b 
   * @returns 
   */
  addition(b: Matrix) {
    return this.coLocationOperation(b, 'add')
  }
  /**
   * 数乘
   * @param b 
   * @returns 
   */
  numberMultiply(b: number) {
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        m.push(this.self[i][j] * b)
      }
      n.push(m)
    }
    return new Matrix(n)
  }
  multiply(b: Matrix) {
    if (this.shape[1] !== b.shape[0]) {
      throw new Error('当矩阵A的列数等于矩阵B的行数，A与B才可以相乘')
    }
    let row = this.shape[0]
    let col = b.shape[1]
    let bt = b.transposition()
    let n = []
    for (let i = 0; i < row; i++) {
      let m = []
      for (let k = 0; k < col; k++) {
        let tm = this.self[i].reduce((p, c, j) => {
          return p + c * bt.self[k][j]
        }, 0)
        m.push(tm)
      }
      n.push(m)
    }
    return new Matrix(n)
  }
  private scale() { }
  /**
   * 转置
   * @returns 
   */
  transposition() {
    let a = []
    for (let i = 0; i < this.shape[1]; i++) {
      let n = []
      for (let j = 0; j < this.shape[0]; j++) {
        n.push(this.self[j][i])
      }
      a.push(n)
    }
    return new Matrix(a)
  }

  /**
   * 归一化  
   * `X = X - average / range`
   * @returns 
   */
  normalization() {
    let t = this.transposition()
    let n = []
    for (let i = 0; i < t.shape[0]; i++) {
      const max = Math.max(...t.self[i])
      const min = Math.min(...t.self[i])
      const range = max - min
      const average = min + (range / 2)
      n.push([average, range])
      for (let j = 0; j < t.shape[1]; j++) {
        t.self[i][j] = range === 0 ? 0 : (t.self[i][j] - average) / range
      }
    }
    //返回两个矩阵：归一化的矩阵、缩放比矩阵
    return [t.transposition(), new Matrix(n).transposition()]
  }
  /**
   * 打印
   */
  print() {
    console.log(`Matrix ${this.shape[0]}x${this.shape[1]} [`)
    for (let i = 0; i < this.shape[0]; i++) {
      let line = ' '
      for (let j = 0; j < this.shape[1]; j++) {
        line += this.self[i][j] + ', '
      }
      console.log(line)
    }
    console.log(']')
  }
}