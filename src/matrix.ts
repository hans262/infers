export interface GenerateMatrixOptions {
  range: [number, number],
  integer?: boolean
}

export class Matrix {
  shape: [number, number]
  private self: number[][]

  constructor(data: number[][]) {
    if (!data[0]) throw new Error('Matrix at least one row')
    let t = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length)
    if (t) throw new Error('Matrix column inconsistent')
    if (!data[0].length) throw new Error('Matrix has at least one element from row')
    this.shape = [data.length, data[0].length]
    this.self = data
  }

  /**
   * 上下分割矩阵
   */
  slice(start: number, end: number) {
    return new Matrix(this.self.slice(start, end))
  }

  /**
   * 返回行的最大值索引
   */
  argMax(row: number) {
    let d = this.getRow(row)
    let max = d[0]
    let index = 0
    for (let i = 0; i < d.length; i++) {
      if (d[i] > max) {
        max = d[i]
        index = i
      }
    }
    return index
  }

  /**
   * 连接两个矩阵
   * 从底部连接
   */
  connect(b: Matrix) {
    if (this.shape[1] !== b.shape[1]) {
      throw new Error('Matrix column inconsistent')
    }
    let tmp = this.dataSync().concat(b.dataSync())
    return new Matrix(tmp)
  }

  /**
   * 返回新的归零矩阵
   */
  zeroed() {
    return this.atomicOperation(_ => 0)
  }

  /**
   * 克隆当前矩阵
   */
  clone() {
    return new Matrix(this.dataSync())
  }

  /**
   * 获取某一行的均值
   * @param row
   */
  getMeanOfRow(row: number) {
    let tmp = this.getRow(row)
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * 矩阵所有值求和
   */
  sum() {
    let s = 0
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        s += this.get(i, j)
      }
    }
    return s
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
   * 生成nxn单位矩阵
   * @param row 
   */
  static generateIdentity(row: number) {
    let t = this.generate(row, row, 0)
    let col = 0
    for (let n = 0; n < t.shape[0]; n++) {
      t.update(n, col++, 1)
    }
    return t
  }

  /**
   * 生成矩阵
   * @param row 
   * @param col 
   * @param opt //默认为-0.5 ~ 0.5随机值
   */
  static generate(row: number, col: number, opt: GenerateMatrixOptions | number = { range: [-0.5, 0.5] }) {
    let n = []
    for (let i = 0; i < row; i++) {
      let m = []
      for (let j = 0; j < col; j++) {
        let v = 0
        if (typeof opt === 'number') {
          v = opt
        } else {
          let [min, max] = [Math.min(...opt.range), Math.max(...opt.range)]
          let b = min < 0 || max < 0 ? -1 : 0
          v = Math.random() * (max - min) + min + b
          if (opt.integer) {
            v = ~~v
          }
        }
        m.push(v)
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
   * @param oper 
   */
  update(row: number, col: number, val: number, oper?: '+=' | '-=' | '*=' | '/=') {
    switch (oper) {
      case '+=':
        this.self[row][col] += val
        break
      case '-=':
        this.self[row][col] -= val
        break
      case '*=':
        this.self[row][col] *= val
        break
      case '/=':
        this.self[row][col] /= val
        break
      default:
        this.self[row][col] = val
    }
  }

  /**
   * 向矩阵上下左右追加一列/行
   * @param n 
   * @param position 
   */
  expand(n: number, position: 'L' | 'R' | 'T' | 'B') {
    let m: number[][] = []
    for (let i = 0; i < this.shape[0]; i++) {
      let rows = position === 'L' ? [n, ...this.getRow(i)] :
        position === 'R' ? [...this.getRow(i), n] : [...this.getRow(i)]
      m.push(rows)
    }

    if (position === 'T') {
      m.unshift(new Array<number>(m[0].length).fill(n))
    } else if (position === 'B') {
      m.push(new Array<number>(m[0].length).fill(n))
    }

    return new Matrix(m)
  }

  /**
   * 根据索引获取元素
   * @param row 
   * @param col 
   */
  get(row: number, col: number) {
    return this.self[row][col]
  }

  /**
   * 根据索引获取行
   * @param row 
   */
  getRow(row: number) {
    return [...this.self[row]]
  }

  /**
   * 根据索引获取列
   * @param col 
   */
  getCol(col: number) {
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      for (let j = 0; j < this.shape[1]; j++) {
        if (j === col) {
          n.push(this.get(i, j))
        }
      }
    }
    return n
  }

  /**
   * 伴随矩阵
   * - n阶：A(i,j) = (-1)^(i+j) * M(i,j)
   * - 2阶：主对角线元素互换，副对角线元素变号
   * - 1阶：伴随矩阵为一阶单位方阵
   */
  adjugate() {
    if (this.shape[0] !== this.shape[1]) throw new Error('只有方阵才能求伴随矩阵')
    if (this.shape[0] === 1) return new Matrix([[1]])
    if (this.shape[0] === 2) {
      return new Matrix([
        [this.get(1, 1), this.get(0, 1) * -1],
        [this.get(1, 0) * -1, this.get(0, 0)]
      ])
    }
    return this.clone().atomicOperation((_, r, c) =>
      this.cominor(r, c).det() * ((-1) ** (r + c + 2))
    ).T
  }

  /**
   * 矩阵的逆  
   * 克拉默法则: A-1 = adjA / detA
   */
  inverse() {
    if (this.shape[0] !== this.shape[1]) throw new Error('只有方阵才能求逆')
    let det = this.det()
    if (det === 0) throw new Error('该矩阵不可逆')
    let ad = this.adjugate()
    return ad.atomicOperation(item => item / det)
  }

  /**
   * 行列式计算
   */
  det() {
    if (this.shape[0] !== this.shape[1]) throw new Error('只有方阵才能计算行列式')
    if (this.shape[0] === 1) throw new Error('矩阵行必须大于1')
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
   * 筛选余子式矩阵
   * @param row 行
   * @param col 列
   */
  cominor(row: number, col: number) {
    if (this.shape[0] < 2 || this.shape[1] < 2) {
      throw new Error('求余子式行和列必须大于2才有意义')
    }
    let n = this.dataSync().map((v) => {
      v = v.filter((_, j) => j !== col)
      return v
    }).filter((_, i) => i !== row)
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
  coLocationOperation(b: Matrix, oper: 'add' | 'sub' | 'mul' | 'exp') {
    if (!this.equalsShape(b)) {
      throw new Error('必须满足两个矩阵是同形矩阵')
    }
    let n = []
    for (let i = 0; i < this.shape[0]; i++) {
      let m = []
      for (let j = 0; j < this.shape[1]; j++) {
        let [x, y] = [this.get(i, j), b.get(i, j)]
        let c = oper === 'add' ? x + y :
          oper === 'sub' ? x - y :
            oper === 'mul' ? x * y :
              oper === 'exp' ? x / y : x
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
   * 矩阵乘法 ｜ 数乘
   * @param b 
   */
  multiply(b: Matrix | number) {
    if (typeof b === 'number') {
      return this.atomicOperation(item => item * b)
    }
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
   * 专置
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
   * 特征缩放  
   * - X' = X - average / range  -0.5 ~ 0.5
   * - X' = X - min / range  0 ~ 1
   * @returns [归一化矩阵, 缩放比矩阵]
   */
  normalization(type: 'average' | 'min' = 'average') {
    let t = this.T
    let n = []
    for (let i = 0; i < t.shape[0]; i++) {
      let max = Math.max(...t.getRow(i))
      let min = Math.min(...t.getRow(i))
      let range = max - min
      let average = min + (range / 2)
      let temp = type === 'average' ? average : min
      n.push([temp, range])
      for (let j = 0; j < t.shape[1]; j++) {
        let s = range === 0 ? 0 : (t.get(i, j) - temp) / range
        t.update(i, j, s)
      }
    }
    return [t.T, new Matrix(n).T]
  }

  /**
   * 格式化输出
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