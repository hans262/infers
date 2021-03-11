"use strict";
function drawPath(ctx, path) {
    const firstPoint = path[0];
    ctx.beginPath();
    ctx.moveTo(firstPoint[0], firstPoint[1]);
    path.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.stroke();
}
function drawPoint(ctx, point) {
    ctx.beginPath();
    ctx.fillStyle = 'red';
    ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
    ctx.fill();
}
function main() {
    const canvas = document.querySelector('#canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 800;
    canvas.height = 400;
    drawPath(ctx, [
        [100, 90], [300, 80],
        [350, 200], [250, 180],
        [180, 270], [100, 190]
    ]);
}
main();
//# sourceMappingURL=drawTest.js.map