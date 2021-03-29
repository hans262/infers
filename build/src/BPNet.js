"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BPNet = void 0;
const matrix_1 = require("./matrix");
class BPNet {
    constructor(shape, netconf) {
        this.shape = shape;
        this.netconf = netconf;
        this.rate = 0.001;
        if (shape.length < 2) {
            throw new Error('The network has at least two layers');
        }
        this.nlayer = shape.length;
        const [w, b] = this.initwb();
        this.w = w;
        this.b = b;
    }
    nOfLayer(l) {
        let n = this.shape[l];
        return Array.isArray(n) ? n[0] : n;
    }
    afOfLayer(l) {
        let n = this.shape[l];
        return Array.isArray(n) ? n[1] : undefined;
    }
    initwb(v) {
        let w = [];
        let b = [];
        for (let l = 1; l < this.shape.length; l++) {
            w[l] = matrix_1.Matrix.generate(this.nOfLayer(l), this.nOfLayer(l - 1), v);
            b[l] = matrix_1.Matrix.generate(1, this.nOfLayer(l), v);
        }
        return [w, b];
    }
    setRate(rate) {
        this.rate = rate;
    }
    afn(x, l) {
        let af = this.afOfLayer(l);
        switch (af) {
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
    afd(x, l) {
        let af = this.afOfLayer(l);
        switch (af) {
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
        let hy = [];
        for (let l = 0; l < this.nlayer; l++) {
            if (l === 0) {
                hy[l] = xs;
                continue;
            }
            hy[l] = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) => this.afn(item + this.b[l].get(0, j), l));
        }
        return hy;
    }
    zoomScalem(xs) {
        return xs.atomicOperation((item, _, j) => {
            if (!this.scalem)
                return item;
            return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j);
        });
    }
    predict(xs) {
        if (xs.shape[1] !== this.nOfLayer(0)) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`);
        }
        return this.calcnet(this.zoomScalem(xs));
    }
    calcDerivative(hy, ys, n) {
        const [dw, dy] = this.initwb(0);
        for (let l = this.nlayer - 1; l > 0; l--) {
            if (l === this.nlayer - 1) {
                for (let j = 0; j < this.nOfLayer(l); j++) {
                    dy[l].update(0, j, (hy[l].get(n, j) - ys.get(n, j)) * this.afd(hy[l].get(n, j), l));
                    for (let k = 0; k < this.nOfLayer(l - 1); k++) {
                        dw[l].update(j, k, hy[l - 1].get(n, k) * dy[l].get(0, j));
                    }
                }
                continue;
            }
            for (let j = 0; j < this.nOfLayer(l); j++) {
                for (let i = 0; i < this.nOfLayer(l + 1); i++) {
                    dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=');
                }
                dy[l].update(0, j, this.afd(hy[l].get(n, j), l), '*=');
                for (let k = 0; k < this.nOfLayer(l - 1); k++) {
                    dw[l].update(j, k, hy[l - 1].get(n, k) * dy[l].get(0, j));
                }
            }
        }
        return { dy, dw };
    }
    update(dy, dw) {
        for (let l = 1; l < this.nlayer; l++) {
            this.w[l] = this.w[l].subtraction(dw[l].numberMultiply(this.rate));
            this.b[l] = this.b[l].subtraction(dy[l].numberMultiply(this.rate));
        }
    }
    cost(hy, ys) {
        let m = ys.shape[0];
        let sub = hy[this.nlayer - 1].subtraction(ys).atomicOperation(item => item ** 2).columnSum();
        let tmp = sub.getRow(0).map(v => (1 / (2 * m)) * v);
        return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    crossCost(hy, ys) {
        let m = ys.shape[0];
        let t = hy[this.nlayer - 1].atomicOperation((h, i, j) => {
            let y = ys.get(i, j);
            return y === 1 ? -Math.log(h) : -Math.log(1 - h);
        }).columnSum();
        let tmp = t.getRow(0).map(v => (1 / m) * v);
        return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    bgd(hy, ys) {
        let m = ys.shape[0];
        const [ndw, ndy] = this.initwb(0);
        for (let n = 0; n < m; n++) {
            const { dy, dw } = this.calcDerivative(hy, ys, n);
            for (let l = 1; l < this.nlayer; l++) {
                ndw[l] = ndw[l].addition(dw[l]);
                ndy[l] = ndy[l].addition(dy[l]);
            }
        }
        for (let l = 1; l < this.nlayer; l++) {
            ndw[l] = ndw[l].atomicOperation(item => item / m);
            ndy[l] = ndy[l].atomicOperation(item => item / m);
        }
        this.update(ndy, ndw);
    }
    sgd(hy, ys) {
        let m = ys.shape[0];
        for (let n = 0; n < m; n++) {
            const { dy, dw } = this.calcDerivative(hy, ys, n);
            this.update(dy, dw);
        }
    }
    fit(xs, ys, batch, callback) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('输入输出矩阵行数不统一');
        }
        if (xs.shape[1] !== this.nOfLayer(0)) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`);
        }
        if (ys.shape[1] !== this.nOfLayer(this.nlayer - 1)) {
            throw new Error(`标签与网络输出不符合，output num -> ${this.nOfLayer(this.nlayer - 1)}`);
        }
        const [nxs, scalem] = xs.normalization();
        this.scalem = scalem;
        xs = nxs;
        for (let p = 0; p < batch; p++) {
            let hy = this.calcnet(xs);
            if (this.netconf && this.netconf.optimizer === 'BGD') {
                this.bgd(hy, ys);
            }
            else {
                this.sgd(hy, ys);
            }
            let loss = this.cost(hy, ys);
            if (callback)
                callback(p, loss);
        }
    }
}
exports.BPNet = BPNet;
//# sourceMappingURL=BPNet.js.map