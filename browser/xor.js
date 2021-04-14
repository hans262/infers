import { BPNet, Matrix, toFixed } from './infers.esm.js'

const [W, H] = [1200, 600]

const canvas = document.getElementById('canvas')
canvas.width = W
canvas.height = H
canvas.style.border = '1px solid'
const ctx = canvas.getContext('2d')

function drawCircle(x, y, r, n) {
  ctx.beginPath()
  ctx.arc(x, y, r, 0, 2 * Math.PI)
  ctx.fillStyle = 'white'
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = 'blue'
  ctx.font = "30px Verdana";
  ctx.fillText(n, x - 10, y + 12)
}

function drawLine(x1, y1, x2, y2) {
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
}

function drawText(epoch, loss) {
  ctx.beginPath()
  ctx.fillStyle = 'blue'
  ctx.font = "30px Verdana";
  let n = 'epoch: ' + epoch + ', loss: ' + loss
  ctx.fillText(n, 100, 520)
}

function drawNet(model, xs) {
  let xs2 = model.scaled(new Matrix([xs]))
  let hy = model.calcnet(xs2)
  ctx.fillStyle = "white"
  ctx.fillRect(0, 0, W, H)
  let max = Math.max(...model.shape.map(v => Array.isArray(v) ? v[0] : v))
  let left = 50
  let top = 70
  let wx = 300
  for (let l = 0; l < model.nlayer; l++) {
    let n = model.unit(l)
    let pu = (max - n) / 2
    let n2 = model.unit(l + 1)
    let pu2 = (max - n2) / 2
    for (let j = 0; j < n; j++) {
      let x = left + wx * l
      let y = top + 70 * j + pu * 70
      for (let k = 0; k < n2; k++) {
        let x2 = x + wx
        let y2 = 70 + 70 * k + pu2 * 70
        drawLine(x, y, x2, y2)
      }
      drawCircle(x, y, 20, toFixed(hy[l].get(0, j), 5))
    }
  }
}

let model = new BPNet([2, [6, 'Tanh'], [6, 'Tanh'], [2, 'Sigmoid']], { rate: 0.1 })
drawNet(model, [1, 0])

let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])

let buts = document.querySelectorAll('#buts button')

buts.forEach(but => {
  but.addEventListener('click', () => {
    if (but.innerHTML === 'TRAIN') {
      but.disabled = true
      model.fit(xs, ys, {
        epochs: 1000, async: true, onEpoch: (epoch, loss) => {
          drawNet(model, [1, 0])
          drawText(epoch, loss)
        }, onTrainEnd: () => {
          but.disabled = false
        }
      })
      return
    }
    let tmp = but.innerHTML.split(', ').map(v => Number(v))
    drawNet(model, tmp)
  })
})