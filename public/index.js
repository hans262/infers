let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])

let model = new BPNet([2, [6, 'Tanh'], [2, 'Sigmoid']], { rate: 0.1 })

const canvas = document.getElementById('canvas')
canvas.width = 800
canvas.height = 600
canvas.style.border = '1px solid'
const ctx = canvas.getContext('2d')

function drawArc(x, y, n) {
  ctx.beginPath()
  ctx.arc(x, y, 20, 0, 2 * Math.PI)
  ctx.fillStyle = 'white'
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = 'red'
  ctx.font = "30px Verdana";
  ctx.fillText(n.toString(), x - 10, y + 13)
}

function drawPath(x1, y1, x2, y2) {
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
}

function render() {
  let xs2 = model.zoomScalem(new Matrix([[1, 0]]))
  let res = model.calcnet(xs2)
  ctx.fillStyle = "white"
  ctx.fillRect(0, 0, 800, 600)
  let arr = []
  for (let l = 0; l < model.shape.length; l++) {
    arr.push(model.nOfLayer(l))
  }
  let max = Math.max(...arr)
  let sx = 100
  let wx = 200
  for (let l = 0; l < model.shape.length; l++) {
    let n = model.nOfLayer(l)
    let pu = (max - n) / 2
    let n2 = model.nOfLayer(l + 1)
    let pu2 = (max - n2) / 2
    for (let j = 0; j < n; j++) {
      let x = sx + wx * l
      let y = 70 + 70 * j + pu * 70
      for (let k = 0; k < n2; k++) {
        let x2 = x + wx
        let y2 = 70 + 70 * k + pu2 * 70
        drawPath(x, y, x2, y2)
      }
      drawArc(x, y, toFixed(res[l].get(0, j), 5))
    }
  }
}

model.fit(xs, ys, {
  epochs: 5000, onEpoch: (epoch, loss) => {
    if (epoch % 100 === 0) {
      console.log('epoch = ' + epoch, loss)
    }
  }
})

render()
model.predict(xs).print()