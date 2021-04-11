"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = require("./matrix");
function toFixed(num, fix) {
    const amount = 10 ** fix;
    return ~~(num * amount) / amount;
}
exports.toFixed = toFixed;
function upset(xs, ys) {
    let xss = xs.dataSync();
    let yss = ys.dataSync();
    for (let i = 1; i < ys.shape[0]; i++) {
        let random = Math.floor(Math.random() * (i + 1));
        [xss[i], xss[random]] = [xss[random], xss[i]];
        [yss[i], yss[random]] = [yss[random], yss[i]];
    }
    return { xs: new matrix_1.Matrix(xss), ys: new matrix_1.Matrix(yss) };
}
exports.upset = upset;
//# sourceMappingURL=common.js.map