"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.afd = exports.afn = exports.upset = exports.toFixed = void 0;
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
function afn(x, rows, af) {
    switch (af) {
        case 'Sigmoid':
            return 1 / (1 + Math.exp(-x));
        case 'Relu':
            return x >= 0 ? x : 0;
        case 'Tanh':
            return Math.tanh(x);
        case 'Softmax':
            let d = Math.max(...rows);
            return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0);
        default:
            return x;
    }
}
exports.afn = afn;
function afd(x, af) {
    switch (af) {
        case 'Sigmoid':
            return x * (1 - x);
        case 'Relu':
            return x >= 0 ? 1 : 0;
        case 'Tanh':
            return 1 - Math.tanh(x) ** 2;
        case 'Softmax':
        default:
            return 1;
    }
}
exports.afd = afd;
