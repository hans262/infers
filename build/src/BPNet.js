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
    calcDerivative(hy, ys) {
        const [dw, dy] = this.initwb(0);
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
    bgd(xs, ys, conf) {
        let m = ys.shape[0];
        for (let ep = 0; ep < conf.epochs; ep++) {
            let eploss = 0;
            let dws = this.w.map(w => w.zeroed());
            let dys = this.b.map(b => b.zeroed());
            for (let n = 0; n < m; n++) {
                let xss = new matrix_1.Matrix([xs.getRow(n)]);
                let yss = new matrix_1.Matrix([ys.getRow(n)]);
                let hy = this.calcnet(xss);
                let { dy, dw } = this.calcDerivative(hy, yss);
                for (let l = 1; l < this.nlayer; l++) {
                    dws[l] = dws[l].addition(dw[l]);
                    dys[l] = dys[l].addition(dy[l]);
                }
                let loss = hy[this.nlayer - 1].subtraction(yss)
                    .atomicOperation(item => (item ** 2) / 2)
                    .getMeanOfRow(0);
                eploss += loss;
            }
            for (let l = 1; l < this.nlayer; l++) {
                dws[l] = dws[l].atomicOperation(item => item / m);
                dys[l] = dys[l].atomicOperation(item => item / m);
            }
            this.update(dys, dws);
            if (conf.onEpoch)
                conf.onEpoch(ep, eploss / m);
        }
    }
    sgd(xs, ys, conf) {
        let m = ys.shape[0];
        for (let ep = 0; ep < conf.epochs; ep++) {
            let eploss = 0;
            for (let n = 0; n < m; n++) {
                let xss = new matrix_1.Matrix([xs.getRow(n)]);
                let yss = new matrix_1.Matrix([ys.getRow(n)]);
                let hy = this.calcnet(xss);
                const { dy, dw } = this.calcDerivative(hy, yss);
                this.update(dy, dw);
                let loss = hy[this.nlayer - 1].subtraction(yss)
                    .atomicOperation(item => (item ** 2) / 2)
                    .getMeanOfRow(0);
                eploss += loss;
            }
            if (conf.onEpoch)
                conf.onEpoch(ep, eploss / m);
        }
    }
    mbgd(xs, ys, conf) {
        let batchSize = conf.batchSize ? conf.batchSize : 10;
        let batch = 0;
        let b = 0;
        let m = ys.shape[0];
        let dws = this.w.map(w => w.zeroed());
        let dys = this.b.map(b => b.zeroed());
        for (let ep = 0; ep < conf.epochs; ep++) {
            let eploss = 0;
            for (let n = 0; n < m; n++) {
                b += 1;
                let xss = new matrix_1.Matrix([xs.getRow(n)]);
                let yss = new matrix_1.Matrix([ys.getRow(n)]);
                let hy = this.calcnet(xss);
                let { dy, dw } = this.calcDerivative(hy, yss);
                for (let l = 1; l < this.nlayer; l++) {
                    dws[l] = dws[l].addition(dw[l]);
                    dys[l] = dys[l].addition(dy[l]);
                }
                let loss = hy[this.nlayer - 1].subtraction(yss)
                    .atomicOperation(item => (item ** 2) / 2)
                    .getMeanOfRow(0);
                eploss += loss;
                if (b === batchSize || (ep === conf.epochs - 1 && n === m - 1 && b !== 0)) {
                    batch += 1;
                    for (let l = 1; l < this.nlayer; l++) {
                        dws[l] = dws[l].atomicOperation(item => item / b);
                        dys[l] = dys[l].atomicOperation(item => item / b);
                    }
                    this.update(dys, dws);
                    if (conf.onBatch)
                        conf.onBatch(batch, b, loss);
                    dws = dws.map(d => d.zeroed());
                    dys = dys.map(d => d.zeroed());
                    b = 0;
                }
            }
            if (conf.onEpoch)
                conf.onEpoch(ep, eploss / m);
        }
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
        if (conf.batchSize && conf.batchSize > ys.shape[0] * conf.epochs) {
            throw new Error(`批次大小不能大于 epochs * m`);
        }
        const [nxs, scalem] = xs.normalization();
        this.scalem = scalem;
        xs = nxs;
        let mode = this.netconf ? this.netconf.mode : undefined;
        switch (mode) {
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