type Point = [number, number]

function drawPath(ctx: CanvasRenderingContext2D, path: Point[]) {
  const firstPoint = path[0]
  ctx.beginPath()
  ctx.moveTo(firstPoint[0], firstPoint[1])
  path.slice(1).forEach(p => ctx.lineTo(p[0], p[1]))
  ctx.closePath()
  ctx.stroke()
}

function drawPoint(ctx: CanvasRenderingContext2D, point: Point) {
  ctx.beginPath()
  ctx.fillStyle = 'red'
  ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI)
  ctx.fill()
}

function main() {
  const canvas = document.querySelector<HTMLCanvasElement>('#canvas')!
  const ctx = canvas.getContext('2d')!
  canvas.width = 800
  canvas.height = 400
  // drawPath(ctx, [
  //   [100, 100], [200, 100],
  //   [200, 200], [100, 200]
  // ])
  // drawPoint(ctx, [100, 200])
  // drawPath(ctx, [
  //   [10, 8], [200, 75]
  // ])
  // drawPath(ctx, [
  //   [100, 20], [167, 200]
  // ])
  drawPath(ctx, [
    [100, 90], [300, 80],
    [350, 200], [250, 180],
    [180, 270], [100, 190]
  ])
}

main()