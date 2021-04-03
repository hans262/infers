"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BPNet = void 0;
const matrix_1 = require("./matrix");
class BPNet {
    constructor(shape, conf) {
        this.shape = shape;
        this.mode = 'sgd';
        this.rate = 0.01;
        if (shape.length < 2) {
            throw new Error('The network has at least two layers');
        }
        this.nlayer = shape.length;
        const [w, b] = this.initwb();
        this.w = w;
        this.b = b;
        if (conf) {
            if (conf.mode)
                this.mode = conf.mode;
            if (conf.rate)
                this.rate = conf.rate;
            if (conf.w)
                this.w = conf.w;
            if (conf.b)
                this.b = conf.b;
            if (conf.scalem)
                this.scalem = conf.scalem;
        }
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
    afn(x, l, rows) {
        let af = this.afOfLayer(l);
        switch (af) {
            case 'Sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'Relu':
                return x >= 0 ? x : 0;
            case 'Tanh':
                return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            case 'Softmax':
                let d = Math.max(...rows);
                return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0);
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
            case 'Softmax':
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
            let a = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j));
            hy[l] = a.atomicOperation((item, i) => this.afn(item, l, a.getRow(i)));
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
        return this.calcnet(this.zoomScalem(xs))[this.nlayer - 1];
    }
    calcDerivativeMul(hy, ys) {
        let m = ys.shape[0];
        let dws = null;
        let dys = null;
        for (let n = 0; n < m; n++) {
            let nhy = hy.map(item => new matrix_1.Matrix([item.getRow(n)]));
            let nys = new matrix_1.Matrix([ys.getRow(n)]);
            let { dw, dy } = this.calcDerivative(nhy, nys);
            dws = dws ? dws.map((d, l) => d.addition(dw[l])) : dw;
            dys = dys ? dys.map((d, l) => d.addition(dy[l])) : dy;
        }
        dws = dws.map(d => d.atomicOperation(item => item / m));
        dys = dys.map(d => d.atomicOperation(item => item / m));
        return { dy: dys, dw: dws };
    }
    calcDerivative(hy, ys) {
        let dw = this.w.map(w => w.zeroed());
        let dy = this.b.map(b => b.zeroed());
        for (let l = this.nlayer - 1; l > 0; l--) {
            if (l === this.nlayer - 1) {
                for (let j = 0; j < this.nOfLayer(l); j++) {
                    dy[l].update(0, j, (hy[l].get(0, j) - ys.get(0, j)) * this.afd(hy[l].get(0, j), l));
                    for (let k = 0; k < this.nOfLayer(l - 1); k++) {
                        dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j));
                    }
                }
                continue;
            }
            for (let j = 0; j < this.nOfLayer(l); j++) {
                for (let i = 0; i < this.nOfLayer(l + 1); i++) {
                    dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=');
                }
                dy[l].update(0, j, this.afd(hy[l].get(0, j), l), '*=');
                for (let k = 0; k < this.nOfLayer(l - 1); k++) {
                    dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j));
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
        let sub = hy.subtraction(ys).atomicOperation(item => item ** 2).columnSum();
        let tmp = sub.getRow(0).map(v => v / (2 * m));
        return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    bgd(xs, ys, conf) {
        for (let ep = 0; ep < conf.epochs; ep++) {
            let hy = this.calcnet(xs);
            let { dy, dw } = this.calcDerivativeMul(hy, ys);
            this.update(dy, dw);
            if (conf.onEpoch)
                conf.onEpoch(ep, this.cost(hy[this.nlayer - 1], ys));
        }
    }
    sgd(xs, ys, conf) {
        let m = ys.shape[0];
        for (let ep = 0; ep < conf.epochs; ep++) {
            let hys = null;
            for (let n = 0; n < m; n++) {
                let xss = new matrix_1.Matrix([xs.getRow(n)]);
                let yss = new matrix_1.Matrix([ys.getRow(n)]);
                let hy = this.calcnet(xss);
                const { dy, dw } = this.calcDerivative(hy, yss);
                this.update(dy, dw);
                hys = hys ? hys.connect(hy[this.nlayer - 1]) : hy[this.nlayer - 1];
            }
            if (conf.onEpoch)
                conf.onEpoch(ep, this.cost(hys, ys));
        }
    }
    mbgd(xs, ys, conf) {
        let m = ys.shape[0];
        let batchSize = conf.batchSize ? conf.batchSize : 10;
        let batch = Math.ceil(m / batchSize);
        for (let ep = 0; ep < conf.epochs; ep++) {
            let { xs: xst, ys: yst } = this.upset(xs, ys);
            let eploss = 0;
            for (let b = 0; b < batch; b++) {
                let start = b * batchSize;
                let end = start + batchSize;
                end = end > m ? m : end;
                let size = end - start;
                let xss = xst.slice(start, end);
                let yss = yst.slice(start, end);
                let hy = this.calcnet(xss);
                const { dy, dw } = this.calcDerivative(hy, yss);
                this.update(dy, dw);
                let bloss = this.cost(hy[this.nlayer - 1], yss);
                eploss += bloss;
                if (conf.onBatch)
                    conf.onBatch(b, size, bloss);
            }
            if (conf.onEpoch)
                conf.onEpoch(ep, eploss / batch);
        }
    }
    upset(xs, ys) {
        let xss = xs.dataSync();
        let yss = ys.dataSync();
        for (let i = 1; i < ys.shape[0]; i++) {
            let random = Math.floor(Math.random() * (i + 1));
            [xss[i], xss[random]] = [xss[random], xss[i]];
            [yss[i], yss[random]] = [yss[random], yss[i]];
        }
        return { xs: new matrix_1.Matrix(xss), ys: new matrix_1.Matrix(yss) };
    }
    fit(xs, ys, conf) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('输入输出矩阵行数不统一');
        }
        if (xs.shape[1] !== this.nOfLayer(0)) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`);
        }
        if (ys.shape[1] !== this.nOfLayer(this.nlayer - 1)) {
            throw new Error(`标签与网络输出不符合，output num -> ${this.nOfLayer(this.nlayer - 1)}`);
        }
        if (conf.batchSize && conf.batchSize > ys.shape[0]) {
            throw new Error(`批次大小不能大于样本数`);
        }
        const [nxs, scalem] = xs.normalization();
        this.scalem = scalem;
        xs = nxs;
        switch (this.mode) {
            case 'bgd':
                return this.bgd(xs, ys, conf);
            case 'mbgd':
                return this.mbgd(xs, ys, conf);
            case 'sgd':
            default:
                return this.sgd(xs, ys, conf);
        }
    }
}
exports.BPNet = BPNet;
//# sourceMappingURL=BPNet.js.map