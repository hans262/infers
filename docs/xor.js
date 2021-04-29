import { BPNet, Matrix, toFixed } from './infers.esm.js'

const container = document.getElementById('container')
const canvas = document.getElementById('canvas')
const buts = document.querySelectorAll('#buts button')
const rateNode = document.getElementById('rate')
const modeNode = document.getElementById('mode')

const ctx = canvas.getContext('2d')
let [W, H] = [container.getBoundingClientRect().width, 500]

const model = new BPNet([2, [6, 'Tanh'], [6, 'Tanh'], [2, 'Softmax']], { rate: Number(rateNode.value) })
let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])

rateNode.oninput = v => {
  model.rate = Number(v.target.value)
}

modeNode.onchange = v => {
  model.mode = v.target.value
}

function init() {
  W = container.getBoundingClientRect().width
  canvas.width = W
  canvas.height = H
  drawNet(ctx, model, calcHy(xs))
  drawText(ctx, '--', model.calcLoss(xs, ys))
}

window.onresize = init
init()

buts.forEach(but => {
  but.addEventListener('click', () => {
    if (but.innerHTML === 'TRAIN') {
      but.disabled = true
      let epochs = Number(document.getElementById('epochs').value)
      model.fit(xs, ys, {
        epochs, async: true, onEpoch: (epoch, loss) => {
          drawNet(ctx, model, calcHy(xs))
          drawText(ctx, epoch, loss)
        }, onTrainEnd: () => {
          but.disabled = false
        }
      })
      return
    }
    let tmp = but.innerHTML.split(', ').map(v => Number(v))
    let nxs = new Matrix([tmp])
    drawNet(ctx, model, calcHy(nxs))
    drawText(ctx, '--', model.calcLoss(xs, ys))
  })
})

/**
 * @param {CanvasRenderingContext2D} ctx 
 * @param {number} x 
 * @param {number} y 
 * @param {number} r 
 * @param {string | number} n 
 */
function drawCircle(ctx, x, y, r, n) {
  ctx.beginPath()
  ctx.arc(x, y, r, 0, 2 * Math.PI)
  ctx.fillStyle = 'white'
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = '#222'
  ctx.font = "28px sans-serif";
  ctx.fillText(n, x - 8, y + 11)
}

/**
 * @param {CanvasRenderingContext2D} ctx 
 * @param {{x: number, y: number}} p1 
 * @param {{x: number, y: number}} p2 
 */
function drawLine(ctx, p1, p2) {
  ctx.beginPath()
  ctx.moveTo(p1.x, p1.y)
  ctx.lineTo(p2.x, p2.y)
  ctx.stroke()
}

/**
 * @param {CanvasRenderingContext2D} ctx 
 * @param {number} epoch 
 * @param {number} loss 
 */
function drawText(ctx, epoch, loss) {
  ctx.beginPath()
  ctx.fillStyle = '#222'
  ctx.font = "30px sans-serif";
  let n = 'epoch: ' + epoch + ', loss: ' + loss
  ctx.fillText(n, 100, 480)
}

/**
 * @param {CanvasRenderingContext2D} ctx 
 */
function clearCtx(ctx) {
  ctx.fillStyle = "#f6f6f6"
  ctx.fillRect(0, 0, W, H)
}

/**
 * @param {Matrix} xs 
 * @returns {Matrix[]}
 */
function calcHy(xs) {
  let sxs = model.scaled(xs)
  let hy = model.calcnet(sxs)
  return [xs, ...hy]
}

/**
 * @param {CanvasRenderingContext2D} ctx 
 * @param {BPNet} model 
 * @param {Matrix[]} hy 
 */
function drawNet(ctx, model, hy) {
  clearCtx(ctx)
  let maxUnit = Math.max(...model.shape.map(v => Array.isArray(v) ? v[0] : v))
  let unitw = W / 24
  let [top, left, spaceX, spaceY] = [50, 1 * unitw, 6.5 * unitw, 70]

  let nlayer = model.shape.length
  for (let l = 0; l < nlayer; l++) {
    let unit = model.unit(l - 1)
    let nextUnit = model.unit(l)
    let startY = ((maxUnit - unit) / 2) * spaceY
    let nextStartY = ((maxUnit - nextUnit) / 2) * spaceY
    for (let j = 0; j < unit; j++) {
      let x = left + spaceX * l
      let y = top + spaceY * j + startY
      for (let k = 0; k < nextUnit; k++) {
        let nextX = x + spaceX
        let nextY = top + spaceY * k + nextStartY
        drawLine(ctx, { x, y }, { x: nextX, y: nextY })
      }
      drawCircle(ctx, x, y, 20, toFixed(hy[l].get(0, j), 5))
    }
  }
}