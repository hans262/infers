"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = require("./matrix");
class BPNet {
    constructor(shape, af) {
        this.shape = shape;
        this.af = af;
        this.rate = 0.001;
        if (shape.length < 3) {
            throw new Error('网络至少三层');
        }
        this.nlayer = shape.length;
        const [w, b] = this.initwb(shape);
        this.w = w;
        this.b = b;
    }
    initwb(shape) {
        let w = [];
        let b = [];
        for (let l = 1; l < shape.length; l++) {
            w[l] = matrix_1.Matrix.generate(shape[l], shape[l - 1]);
            b[l] = matrix_1.Matrix.generate(1, shape[l]);
        }
        return [w, b];
    }
    setRate(rate) {
        this.rate = rate;
    }
    afn(x) {
        switch (this.af) {
            case 'Sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'Relu':
                return x >= 0 ? x : 0;
            case 'Tanh':
                return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            default:
                return x;
        }
    }
    afd(x) {
        switch (this.af) {
            case 'Sigmoid':
                return x * (1 - x);
            case 'Relu':
                return x >= 0 ? 1 : 0;
            case 'Tanh':
                return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2;
            default:
                return 1;
        }
    }
    calcnet(xs) {
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`);
        }
        let ys = [];
        for (let l = 0; l < this.nlayer; l++) {
            if (l === 0) {
                ys[l] = xs;
                continue;
            }
            let w = this.w[l].T;
            let b = this.b[l];
            ys[l] = ys[l - 1].multiply(w).atomicOperation((item, _, j) => this.afn(item + b.get(0, j)));
        }
        return ys;
    }
    zoomScalem(xs) {
        return xs.atomicOperation((item, _, j) => {
            if (!this.scalem)
                return item;
            return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j);
        });
    }
    predict(xs) {
        return this.calcnet(this.zoomScalem(xs));
    }
    calcDerivative(ys, hy) {
        let dy = [];
        for (let l = this.nlayer - 1; l > 0; l--) {
            if (l === this.nlayer - 1) {
                let n = [];
                for (let j = 0; j < this.shape[l]; j++) {
                    n[j] = (hy[l].get(0, j) - ys[j]) * this.afd(hy[l].get(0, j));
                }
                dy[l] = n;
                continue;
            }
            let n = [];
            for (let j = 0; j < this.shape[l]; j++) {
                n[j] = 0;
                for (let i = 0; i < this.shape[l + 1]; i++) {
                    n[j] += dy[l + 1][i] * this.w[l + 1].get(i, j);
                }
                n[j] *= this.afd(hy[l].get(0, j));
            }
            dy[l] = n;
        }
        return dy;
    }
    update(dy, hy) {
        for (let l = 1; l < this.nlayer; l++) {
            for (let j = 0; j < this.shape[l]; j++) {
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    this.w[l].update(j, i, this.rate * dy[l][j] * hy[l - 1].get(0, i), '-=');
                    this.b[l].update(0, j, this.rate * dy[l][j], '-=');
                }
            }
        }
    }
    fit(xs, ys, batch, callback) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('输入输出矩阵行数不统一');
        }
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`);
        }
        if (ys.shape[1] !== this.shape[this.nlayer - 1]) {
            throw new Error(`标签与网络输出不符合，output num -> ${this.shape[this.nlayer - 1]}`);
        }
        let [inputs, scalem] = xs.normalization();
        this.scalem = scalem;
        xs = inputs;
        for (let p = 0; p < batch; p++) {
            let loss = 0;
            for (let n = 0; n < xs.shape[0]; n++) {
                let xss = new matrix_1.Matrix([xs.getRow(n)]);
                let hy = this.calcnet(xss);
                let dys = this.calcDerivative(ys.getRow(n), hy);
                this.update(dys, hy);
                let e = 0;
                let l = this.nlayer - 1;
                for (let j = 0; j < this.shape[l]; j++) {
                    e += ((ys.get(n, j) - hy[l].get(0, j)) ** 2);
                }
                loss += e / this.shape[l];
            }
            loss = loss / (2 * xs.shape[0]);
            if (callback)
                callback(p, loss);
        }
    }
}
exports.BPNet = BPNet;
//# sourceMappingURL=BPNet.js.map