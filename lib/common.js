"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadBPNet = exports.upset = exports.toFixed = void 0;
const matrix_1 = require("./matrix");
const BPNet_1 = require("./BPNet");
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
function loadBPNet(modelJson) {
    let tmp = JSON.parse(modelJson);
    let w = tmp.w.map((w) => new matrix_1.Matrix(w));
    let b = tmp.b.map((b) => new matrix_1.Matrix(b));
    let scale = tmp.scale ? new matrix_1.Matrix(tmp.scale) : undefined;
    return new BPNet_1.BPNet(tmp.shape, {
        mode: tmp.mode,
        rate: tmp.mode,
        w, b, scale
    });
}
exports.loadBPNet = loadBPNet;
