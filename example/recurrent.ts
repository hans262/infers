class Solver {
  decay_rate = 0.999
  smooth_eps = 1e-8
  step_cache: any = {}
  step(model: any, learning_rate: number, regc: number, clipval: number) {
    // perform parameter update
    let solver_stats: any = {};
    let num_clipped = 0;
    let num_tot = 0;
    for (let k in model) {
      if (model.hasOwnProperty(k)) {
        let m = model[k]; // mat ref
        if (!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
        let s = this.step_cache[k];
        for (let i = 0, n = m.w.length; i < n; i++) {

          // rmsprop adaptive learning rate
          let mdwi = m.dw[i];
          s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

          // gradient clip
          if (mdwi > clipval) {
            mdwi = clipval;
            num_clipped++;
          }
          if (mdwi < -clipval) {
            mdwi = -clipval;
            num_clipped++;
          }
          num_tot++;

          // update (and regularize)
          m.w[i] += - learning_rate * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
          m.dw[i] = 0; // reset gradients for next iteration
        }
      }
    }
    solver_stats['ratio_clipped'] = num_clipped * 1.0 / num_tot;
    return solver_stats;
  }
}

function initRNN(input_size: number, hidden_sizes: number[], output_size: number) {
  // hidden size should be a list
  let model: any = {};
  for (let d = 0; d < hidden_sizes.length; d++) { // loop over depths
    let prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
    let hidden_size = hidden_sizes[d];
    model['Wxh' + d] = Mat.RandMat(hidden_size, prev_size, 0.08);
    model['Whh' + d] = Mat.RandMat(hidden_size, hidden_size, 0.08);
    model['bhh' + d] = new Mat(hidden_size, 1);
  }
  let last_hidden_size = hidden_sizes[hidden_sizes.length - 1]
  // decoder params
  model['Whd'] = Mat.RandMat(output_size, last_hidden_size, 0.08);
  model['bd'] = new Mat(output_size, 1);
  return model;
}

class Tool {
  //随取整数 范围内
  static randi(a: number, b: number) {
    return Math.floor(Math.random() * (b - a) + a)
  }
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
  dw: number[]
  constructor(
    public n: number, // rows
    public d: number //columns
  ) {
    this.w = this.zeros(n * d)
    this.dw = this.zeros(n * d)
  }
  zeros(n: number) {
    return new Array<number>(n).fill(0)
  }
  static RandMat(n: number, d: number, std: number) {
    let m = new Mat(n, d)
    for (let i = 0, n = m.w.length; i < n; i++) {
      m.w[i] = 2 * std * Math.random() - std
    }
    return m
  }
}

function forwardRNN(G: Graph, model: any, hidden_sizes: number[], x: Mat, prev) {
  let hidden_prevs;
  if (typeof prev.h === 'undefined') {
    hidden_prevs = [];
    for (let d = 0; d < hidden_sizes.length; d++) {
      hidden_prevs.push(new Mat(hidden_sizes[d], 1));
    }
  } else {
    hidden_prevs = prev.h;
  }

  let hidden = [];
  for (let d = 0; d < hidden_sizes.length; d++) {

    let input_vector = d === 0 ? x : hidden[d - 1];
    let hidden_prev = hidden_prevs[d];

    let h0 = G.mul(model['Wxh' + d], input_vector);
    let h1 = G.mul(model['Whh' + d], hidden_prev);
    let hidden_d = G.relu(G.add(G.add(h0, h1), model['bhh' + d]));

    hidden.push(hidden_d);
  }

  // one decoder to outputs at end
  let output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]), model['bd']);

  // return cell memory, hidden representation and output
  return { 'h': hidden, 'o': output };
}

class Graph {
  backprop: any = []
  constructor(public needs_backprop: boolean) { }

  backward() {
    for (let i = this.backprop.length - 1; i >= 0; i--) {
      this.backprop[i](); // tick!
    }
  }
  rowPluck(m: Mat, ix: number) {
    let d = m.d;
    let out = new Mat(d, 1);
    for (let i = 0, n = d; i < n; i++) { out.w[i] = m.w[d * ix + i]; } // copy over the data

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


// 模型参数
let hidden_sizes = [20]; // hidden layers
let letter_size = 5; // 字母大小

// 优化
let regc = 0.000001; // L2 正则惩罚
let learning_rate = 0.001; // learning rate
let clipval = 5.0; // 梯度

// 全局变量初始化
let epoch_size = -1
let input_size = -1
let output_size = -1
let letterToIndex: { [key: string]: number } = {}
let indexToLetter: { [key: number]: string } = {}

let data_sents = [
  '今天天气很好',
  '明天是个好日子',
  '你在哪里',
  '这个地方是哪里',
  '你到底会不会来啊'
]

let temp = Array.from(new Set(data_sents.join('').split('')))
for (let i = 0; i < temp.length; i++) {
  letterToIndex[temp[i]] = i + 1
  indexToLetter[i + 1] = temp[i]
}
input_size = temp.length + 1
output_size = temp.length + 1
epoch_size = data_sents.length

let solver = new Solver()

let model = initRNN(letter_size, hidden_sizes, output_size)
model['Wil'] = Mat.RandMat(input_size, letter_size, 0.08)

function forwardIndex(G: Graph, model, ix, prev) {
  let x = G.rowPluck(model['Wil'], ix)
  return forwardRNN(G, model, hidden_sizes, x, prev)
}

function predict(model) {
  let G = new Graph(false)
  let s = ''
  let prev = {}
  let max_chars_gen = 20
  while (true) {
    if (s.length > max_chars_gen) break
    let preIx = s.length === 0 ? 0 : letterToIndex[s[s.length - 1]]
    prev = forwardIndex(G, model, preIx, prev)
    let probs = Tool.softmax(prev.o)
    let nextIx = Tool.maxi(probs.w)
    if (nextIx === 0) break
    s += indexToLetter[nextIx]
  }
  return s
}

function cost(model, sent: string) {
  let n = sent.length;
  let G = new Graph(true);
  let log2ppl = 0.0;
  let cost = 0.0;
  let prev = {};
  for (let i = -1; i < n; i++) {
    let ix_source = i === -1 ? 0 : letterToIndex[sent[i]]
    let ix_target = i === n - 1 ? 0 : letterToIndex[sent[i + 1]]
    let lh = forwardIndex(G, model, ix_source, prev);
    prev = lh;
    let logprobs = lh.o
    let probs = Tool.softmax(logprobs)
    log2ppl += -Math.log2(probs.w[ix_target])
    cost += -Math.log(probs.w[ix_target])
    logprobs.dw = probs.w;
    logprobs.dw[ix_target] -= 1
  }
  let ppl = Math.pow(2, log2ppl / (n - 1))
  return { G, ppl, cost }
}

async function train() {
  for (let i = 0; i < 1000; i++) {
    let sentix = Tool.randi(0, data_sents.length)
    let sent = data_sents[sentix]
    let cost_struct = cost(model, sent)
    cost_struct.G.backward()
    solver.step(model, learning_rate, regc, clipval)

    if (i % 10 === 0) {
      let epoch = (i / epoch_size).toFixed(2)
      let perplexity = cost_struct.ppl.toFixed(2)
      console.log('epoch: ', epoch, 'perplexity: ', perplexity)
    }
  }
  console.log(predict(model))
}

export function run() {
  train()
}