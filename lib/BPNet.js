"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BPNet = void 0;
const common_1 = require("./common");
const matrix_1 = require("./matrix");
class BPNet {
    constructor(shape, conf) {
        this.shape = shape;
        this.mode = 'sgd';
        this.rate = 0.01;
        this.nlayer = shape.length;
        if (this.nlayer < 2) {
            throw new Error('The network has at least two layers');
        }
        this.w = [];
        this.b = [];
        for (let l = 1; l < this.shape.length; l++) {
            this.w[l] = matrix_1.Matrix.generate(this.unit(l), this.unit(l - 1));
            this.b[l] = matrix_1.Matrix.generate(1, this.unit(l));
        }
        if (conf) {
            if (conf.mode)
                this.mode = conf.mode;
            if (conf.rate)
                this.rate = conf.rate;
            if (conf.w)
                this.w = conf.w;
            if (conf.b)
                this.b = conf.b;
            if (conf.scale)
                this.scale = conf.scale;
        }
    }
    unit(l) {
        let n = this.shape[l];
        return Array.isArray(n) ? n[0] : n;
    }
    af(l) {
        let n = this.shape[l];
        return Array.isArray(n) ? n[1] : undefined;
    }
    afn(x, l, rows) {
        let af = this.af(l);
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
        let af = this.af(l);
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
            let tmp = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j));
            hy[l] = tmp.atomicOperation((item, i) => this.afn(item, l, tmp.getRow(i)));
        }
        return hy;
    }
    scaled(xs) {
        if (!this.scale)
            return xs;
        return xs.atomicOperation((item, _, j) => {
            let scale = this.scale;
            let range = scale.get(1, j);
            let average = scale.get(0, j);
            return range === 0 ? 0 : (item - average) / range;
        });
    }
    predict(xs) {
        if (xs.shape[1] !== this.unit(0)) {
            throw new Error(`Input matrix column number error, input shape -> ${this.unit(0)}.`);
        }
        return this.calcnet(this.scaled(xs))[this.nlayer - 1];
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
                for (let j = 0; j < this.unit(l); j++) {
                    dy[l].update(0, j, (hy[l].get(0, j) - ys.get(0, j)) * this.afd(hy[l].get(0, j), l));
                    for (let k = 0; k < this.unit(l - 1); k++) {
                        dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j));
                    }
                }
                continue;
            }
            for (let j = 0; j < this.unit(l); j++) {
                for (let i = 0; i < this.unit(l + 1); i++) {
                    dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=');
                }
                dy[l].update(0, j, this.afd(hy[l].get(0, j), l), '*=');
                for (let k = 0; k < this.unit(l - 1); k++) {
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
    async bgd(xs, ys, conf) {
        for (let ep = 0; ep < conf.epochs; ep++) {
            let hy = this.calcnet(xs);
            let { dy, dw } = this.calcDerivativeMul(hy, ys);
            this.update(dy, dw);
            if (conf.onEpoch) {
                conf.onEpoch(ep, this.cost(hy[this.nlayer - 1], ys));
                conf.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (conf.onTrainEnd && ep === conf.epochs - 1) {
                conf.onTrainEnd(this.cost(hy[this.nlayer - 1], ys));
            }
        }
    }
    async sgd(xs, ys, conf) {
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
            if (conf.onEpoch) {
                conf.onEpoch(ep, this.cost(hys, ys));
                conf.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (conf.onTrainEnd && ep === conf.epochs - 1) {
                conf.onTrainEnd(this.cost(hys, ys));
            }
        }
    }
    async mbgd(xs, ys, conf) {
        let m = ys.shape[0];
        let batchSize = conf.batchSize ? conf.batchSize : 10;
        let batch = Math.ceil(m / batchSize);
        for (let ep = 0; ep < conf.epochs; ep++) {
            let { xs: xst, ys: yst } = common_1.upset(xs, ys);
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
            if (conf.onEpoch) {
                conf.onEpoch(ep, eploss / batch);
                conf.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (conf.onTrainEnd && ep === conf.epochs - 1) {
                conf.onTrainEnd(eploss / batch);
            }
        }
    }
    fit(xs, ys, conf) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('The row number of input and output matrix is not uniform.');
        }
        if (xs.shape[1] !== this.unit(0)) {
            throw new Error(`Input matrix column number error, input shape -> ${this.unit(0)}.`);
        }
        if (ys.shape[1] !== this.unit(this.nlayer - 1)) {
            throw new Error(`Output matrix column number error, output shape -> ${this.unit(this.nlayer - 1)}.`);
        }
        if (conf.batchSize && conf.batchSize > ys.shape[0]) {
            throw new Error(`The batch size cannot be greater than the number of samples.`);
        }
        const [nxs, scale] = xs.normalization();
        this.scale = scale;
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
