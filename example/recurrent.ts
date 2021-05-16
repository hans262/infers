class Tool {
  //返回数组中最大值的索引
  static maxi(w: number[]) {
    let maxv = w[0]
    let maxix = 0
    for (let i = 1; i < w.length; i++) {
      let v = w[i]
      if (v > maxv) {
        maxix = i
        maxv = v
      }
    }
    return maxix;
  }

  static softmax(m: Mat) {
    let out = new Mat(m.n, m.d); // probability volume
    let maxval = -999999;
    for (let i = 0, n = m.w.length; i < n; i++) { if (m.w[i] > maxval) maxval = m.w[i]; }

    let s = 0.0;
    for (let i = 0, n = m.w.length; i < n; i++) {
      out.w[i] = Math.exp(m.w[i] - maxval);
      s += out.w[i];
    }
    for (let i = 0, n = m.w.length; i < n; i++) { out.w[i] /= s; }

    return out;
  }
}

class Mat {
  w: number[]
  dw: number[] //对应求导
  constructor(
    public n: number, // rows
    public d: number //columns
  ) {
    this.w = new Array<number>(n * d).fill(0)
    this.dw = new Array<number>(n * d).fill(0)
  }

  static RandMat(n: number, d: number, std: number) {
    let m = new Mat(n, d)
    for (let i = 0; i < m.w.length; i++) {
      m.w[i] = 2 * std * Math.random() - std
    }
    return m
  }
}

function forwardRNN(G: Graph, model: Model, hidden_size: number, x: Mat, prev_hidden?: Mat) {
  prev_hidden = prev_hidden ? prev_hidden : new Mat(hidden_size, 1)

  let input_vector = x
  let hidden_prev = prev_hidden

  let h0 = G.mul(model['Wxh'], input_vector)
  let h1 = G.mul(model['Whh'], hidden_prev)

  let hidden = G.relu(G.add(G.add(h0, h1), model['bhh']))

  let output = G.add(G.mul(model['Whd'], hidden), model['bd']);
  return { 'h': hidden, 'o': output };
}

class Graph {
  backprop: (() => void)[] = []
  constructor(public needs_backprop: boolean) { }

  backward() {
    for (let i = this.backprop.length - 1; i >= 0; i--) {
      this.backprop[i]()
    }
  }

  rowPluck(m: Mat, ix: number) {
    let d = m.d;
    let out = new Mat(d, 1)
    for (let i = 0, n = d; i < n; i++) { out.w[i] = m.w[d * ix + i]; } // copy

    if (this.needs_backprop) {
      let backward = function () {
        for (let i = 0, n = d; i < n; i++) { m.dw[d * ix + i] += out.dw[i]; }
      }
      this.backprop.push(backward);
    }
    return out;
  }

  relu(m: Mat) {
    let out = new Mat(m.n, m.d);
    let n = m.w.length;
    for (let i = 0; i < n; i++) {
      out.w[i] = Math.max(0, m.w[i]); // relu
    }
    if (this.needs_backprop) {
      let backward = function () {
        for (let i = 0; i < n; i++) {
          m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
        }
      }
      this.backprop.push(backward);
    }
    return out;
  }

  mul(m1: Mat, m2: Mat) {
    let n = m1.n;
    let d = m2.d;
    let out = new Mat(n, d);
    for (let i = 0; i < m1.n; i++) { // loop over rows of m1
      for (let j = 0; j < m2.d; j++) { // loop over cols of m2
        let dot = 0.0;
        for (let k = 0; k < m1.d; k++) { // dot product loop
          dot += m1.w[m1.d * i + k] * m2.w[m2.d * k + j];
        }
        out.w[d * i + j] = dot;
      }
    }

    if (this.needs_backprop) {
      let backward = function () {
        for (let i = 0; i < m1.n; i++) { // loop over rows of m1
          for (let j = 0; j < m2.d; j++) { // loop over cols of m2
            for (let k = 0; k < m1.d; k++) { // dot product loop
              let b = out.dw[d * i + j];
              m1.dw[m1.d * i + k] += m2.w[m2.d * k + j] * b;
              m2.dw[m2.d * k + j] += m1.w[m1.d * i + k] * b;
            }
          }
        }
      }
      this.backprop.push(backward);
    }
    return out;
  }

  add(m1: Mat, m2: Mat) {
    let out = new Mat(m1.n, m1.d);
    for (let i = 0, n = m1.w.length; i < n; i++) {
      out.w[i] = m1.w[i] + m2.w[i];
    }
    if (this.needs_backprop) {
      let backward = function () {
        for (let i = 0, n = m1.w.length; i < n; i++) {
          m1.dw[i] += out.dw[i];
          m2.dw[i] += out.dw[i];
        }
      }
      this.backprop.push(backward);
    }
    return out;
  }
}

interface Model {
  Wil: Mat

  Wxh: Mat
  Whh: Mat
  bhh: Mat

  Whd: Mat
  bd: Mat
}

class RNN {
  hidden_size = 20; // hidden layers
  letter_size = 5; // 字母大小
  learning_rate = 0.02 // learning rate

  input_size: number
  output_size: number

  letterToIndex: { [key: string]: number } = {}
  indexToLetter: { [key: number]: string } = {}

  model: Model
  tarin_data: string[]

  constructor(data: string[]) {
    this.tarin_data = data

    let temp = Array.from(new Set(this.tarin_data.join('').split('')))
    for (let i = 0; i < temp.length; i++) {
      this.letterToIndex[temp[i]] = i + 1
      this.indexToLetter[i + 1] = temp[i]
    }
    this.input_size = temp.length + 1
    this.output_size = temp.length + 1

    this.model = this.initRNN(this.input_size, this.letter_size, this.hidden_size, this.output_size)
  }

  initRNN(input_size: number, letter_size: number, hidden_size: number, output_size: number): Model {
    return {
      Wil: Mat.RandMat(input_size, letter_size, 0.08),

      Wxh: Mat.RandMat(hidden_size, letter_size, 0.08),
      Whh: Mat.RandMat(hidden_size, hidden_size, 0.08),
      bhh: new Mat(hidden_size, 1),

      Whd: Mat.RandMat(output_size, hidden_size, 0.08),
      bd: new Mat(output_size, 1),
    }
  }

  forwardIndex(G: Graph, ix: number, prevh?: Mat) {
    let x = G.rowPluck(this.model.Wil, ix)
    return forwardRNN(G, this.model, this.hidden_size, x, prevh)
  }

  predict() {
    let graph = new Graph(false)
    let s = ''
    let prevh: Mat | undefined = undefined;
    let max_chars_gen = 20
    while (true) {
      if (s.length > max_chars_gen) break
      let preIx = s.length === 0 ? 0 : this.letterToIndex[s[s.length - 1]]
      let hy = this.forwardIndex(graph, preIx, prevh)
      prevh = hy.h
      let probs = Tool.softmax(hy.o)
      let nextIx = Tool.maxi(probs.w)
      if (nextIx === 0) break
      s += this.indexToLetter[nextIx]
    }
    return s
  }

  cost(sent: string) {
    let n = sent.length
    let graph = new Graph(true)
    let cost = 0
    let prevh: Mat | undefined = undefined

    for (let i = -1; i < n; i++) {
      let ix_source = i === -1 ? 0 : this.letterToIndex[sent[i]]
      let ix_target = i === n - 1 ? 0 : this.letterToIndex[sent[i + 1]]
      let res = this.forwardIndex(graph, ix_source, prevh)
      prevh = res.h

      let probs = Tool.softmax(res.o)
      cost += -Math.log(probs.w[ix_target])
      res.o.dw = probs.w
      res.o.dw[ix_target] -= 1
    }
    return { graph, cost }
  }

  adjust() {
    let k: keyof Model
    for (k in this.model) {
      let m = this.model[k]
      for (let i = 0; i < m.w.length; i++) {
        m.w[i] -= this.learning_rate * m.dw[i]
        m.dw[i] = 0 //重置导数
      }
    }
  }

  train() {
    for (let i = 0; i < 2000; i++) {
      let cost = 0
      for (let n = 0; n < this.tarin_data.length; n++) {
        let sent = this.tarin_data[n]
        let cost_struct = this.cost(sent)
        cost_struct.graph.backward()
        this.adjust()
        cost += cost_struct.cost
      }
      if (i % 10 === 0) {
        console.log('epoch: ', i, 'cost: ', cost / this.tarin_data.length)
      }
    }
  }
}

export function run() {
  let data = [
    '今天天气很好',
    '明天是个好日子',
    '你在哪里',
    '这个地方是哪里',
    '你到底会不会来啊'
  ]
  let net = new RNN(data)
  net.train()
  console.log(net.predict())
}