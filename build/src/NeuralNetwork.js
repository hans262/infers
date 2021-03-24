"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = require("./matrix");
class NeuralNetwork {
    constructor(shape) {
        this.w = [];
        this.b = [];
        this.rate = 0.5;
        this.esp = 0.0001;
        if (shape.length < 3) {
            throw new Error('网络至少三层');
        }
        if (shape[0] < 2) {
            throw new Error('输入层至少两个特征');
        }
        this.shape = shape;
        this.layerNum = shape.length;
        for (let l = 1; l < this.layerNum; l++) {
            let witem = [];
            let bitem = [];
            for (let j = 0; j < shape[l]; j++) {
                let n = [];
                for (let i = 0; i < shape[l - 1]; i++) {
                    n.push(0.5 - Math.random());
                }
                witem.push(n);
                bitem.push(0.5 - Math.random());
            }
            this.w[l] = witem;
            this.b[l] = bitem;
        }
    }
    setRate(rate) {
        this.rate = rate;
    }
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    predict(xs) {
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`);
        }
        let ys = [];
        for (let l = 0; l < this.layerNum; l++) {
            if (l === 0) {
                ys[l] = xs;
                continue;
            }
            let w = new matrix_1.Matrix(this.w[l]).T;
            let b = new matrix_1.Matrix([this.b[l]]);
            ys[l] = ys[l - 1].multiply(w).atomicOperation((item, _, j) => this.sigmoid(item + b.get(0, j)));
        }
        return ys;
    }
    calcNetwork(xs) {
        let ys = [];
        for (let l = 0; l < this.layerNum; l++) {
            if (l === 0) {
                ys[l] = xs;
                continue;
            }
            ys[l] = [];
            for (let j = 0; j < this.shape[l]; j++) {
                let u = 0;
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    u += this.w[l][j][i] * ys[l - 1][i];
                }
                u += this.b[l][j];
                ys[l][j] = this.sigmoid(u);
            }
        }
        return ys;
    }
    calcdelta(ys, hy) {
        let delta = [];
        for (let l = this.layerNum - 1; l > 0; l--) {
            if (l === this.layerNum - 1) {
                let n = [];
                for (let j = 0; j < this.shape[l]; j++) {
                    n[j] = (ys[j] - hy[l][j]) * hy[l][j] * (1 - hy[l][j]);
                }
                delta[l] = n;
                continue;
            }
            let n = [];
            for (let j = 0; j < this.shape[l]; j++) {
                n[j] = 0;
                for (let i = 0; i < this.shape[l + 1]; i++) {
                    n[j] += delta[l + 1][i] * this.w[l + 1][i][j];
                }
                n[j] *= hy[l][j] * (1 - hy[l][j]);
            }
            delta[l] = n;
        }
        return delta;
    }
    update(hy, delta) {
        for (let l = 1; l < this.layerNum; l++) {
            for (let j = 0; j < this.shape[l]; j++) {
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    this.w[l][j][i] += this.rate * delta[l][j] * hy[l - 1][i];
                    this.b[l][j] += this.rate * delta[l][j];
                }
            }
        }
    }
    fit(xs, ys, batch, callback) {
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`);
        }
        if (ys.shape[1] !== this.shape[this.layerNum - 1]) {
            throw new Error(`标签与网络输出不符合，output num -> ${this.shape[this.layerNum - 1]}`);
        }
        for (let p = 0; p < batch; p++) {
            let loss = 0;
            for (let i = 0; i < xs.shape[0]; i++) {
                let hy = this.calcNetwork(xs.getRow(i));
                let delta = this.calcdelta(ys.getRow(i), hy);
                this.update(hy, delta);
                let n = 0;
                let l1 = this.layerNum - 1;
                for (let l = 0; l < this.shape[l1]; l++) {
                    n += ((ys.get(i, l) - hy[l1][l]) ** 2);
                }
                loss += n / this.shape[l1];
            }
            loss = loss / (2 * xs.shape[0]);
            if (callback)
                callback(p, loss);
            if (loss < this.esp)
                break;
        }
    }
}
exports.NeuralNetwork = NeuralNetwork;
//# sourceMappingURL=NeuralNetwork.js.map