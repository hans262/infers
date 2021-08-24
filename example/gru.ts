var W_z = Math.random() - 0.5
var U_z = Math.random() - 0.5
var bz = Math.random() - 0.5

var W_r = Math.random() - 0.5
var U_r = Math.random() - 0.5
var br = Math.random() - 0.5

var W_h = Math.random() - 0.5
var U_h = Math.random() - 0.5
var bh = Math.random() - 0.5

function forwardPropagation(xs: number[]) {
  var l = xs.length
  var z = new Array<number>(l).fill(0)
  var r = new Array<number>(l).fill(0)
  var h = new Array<number>(l).fill(0)
  var ys = new Array<number>(l).fill(0)

  for (var j = 0; j < l - 1; j++) {
    z[j] = W_z * xs[j] + U_z * ys[j] + bz
    z[j] = 1 / (1 + Math.exp(-1 * z[j]))

    r[j] = W_r * xs[j] + U_r * ys[j] + br
    r[j] = 1 / (1 + Math.exp(-1 * r[j]))

    h[j] = W_h * xs[j] + U_h * ys[j] * r[j] + bh
    h[j] = Math.tanh(h[j])

    ys[j + 1] = (1 - z[j]) * ys[j] + z[j] * h[j]
  }
  return { xs, z, r, h, ys }
}

function backPropagation(data: ReturnType<typeof forwardPropagation>) {
  let { xs, z, r, h, ys } = data
  var l = xs.length
  var rate = 0.02
  var delta = 0

  for (var j = 0; j < l - 1; j++) {
    for (var t = 0; t < j + 1; t++) {
      var d = j - t
      if (t == 0) {
        delta = (ys[d + 1] - xs[d + 1])
      } else {
        delta = (z[d + 1] * (1 - h[d + 1] * h[d + 1]) * (U_h * r[d + 1] + U_h * ys[d + 1] * r[d + 1] * (1 - r[d + 1]) * U_r) + h[d + 1] * z[d + 1] * (1 - z[d + 1]) * U_z + (1 - z[d + 1]) + ys[d + 1] * -1 * z[d + 1] * (1 - z[d + 1]) * U_z) * delta
      }

      W_z = W_z - rate * (h[d] - ys[d]) * z[d] * (1 - z[d]) * delta * xs[d]
      U_z = U_z - rate * (h[d] - ys[d]) * z[d] * (1 - z[d]) * delta * ys[d]
      bz = bz - rate * (h[d] - ys[d]) * z[d] * (1 - z[d]) * delta

      W_r = W_r - rate * z[d] * (1 - h[d] * h[d]) * U_h * ys[d] * r[d] * (1 - r[d]) * delta * xs[d]
      U_r = U_r - rate * z[d] * (1 - h[d] * h[d]) * U_h * ys[d] * r[d] * (1 - r[d]) * delta * ys[d]
      br = br - rate * z[d] * (1 - h[d] * h[d]) * U_h * ys[d] * r[d] * (1 - r[d]) * delta

      W_h = W_h - rate * z[d] * (1 - h[d] * h[d]) * delta * xs[d]
      U_h = U_h - rate * z[d] * (1 - h[d] * h[d]) * delta * ys[d]
      bh = bh - rate * z[d] * (1 - h[d] * h[d]) * delta
    }
  }
}

function train() {
  var xs = [0, 1, 1, 1, 0, 1, 0, 1, 0, 1]

  for (let i = 0; i < 50000; i++) {
    let data = forwardPropagation(xs)
    backPropagation(data)
    let ys = data.ys
    if (i % 100 === 0) {
      let loss = calcLoss(xs, ys)
      console.log('enpoch: ', i, 'loss: ', loss)
    }
  }

  let data = forwardPropagation(xs)
  let ys = integer(data.ys)
  console.log('ys: ', ys)
}

function calcLoss(xs: number[], ys: number[]) {
  let loss = 0
  for (var j = 0; j < xs.length; j++) {
    loss += ((ys[j] - xs[j]) ** 2) / 2
  }
  return loss
}

function integer(ys: number[]) {
  return ys.map(v => v > 0.5 ? 1 : 0)
}

export function gru() {
  train()
}