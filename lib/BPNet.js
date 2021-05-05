"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BPNet = exports.defaultTrainingOptions = void 0;
const common_1 = require("./common");
const matrix_1 = require("./matrix");
const defaultTrainingOptions = (m) => ({
    epochs: 100,
    batchSize: m > 10 ? 10 : m,
    async: false
});
exports.defaultTrainingOptions = defaultTrainingOptions;
class BPNet {
    constructor(shape, opt = {}) {
        this.shape = shape;
        this.mode = 'sgd';
        this.rate = 0.01;
        this.hlayer = shape.length - 1;
        if (this.hlayer < 1) {
            throw new Error('The network has at least two layers');
        }
        this.w = [];
        this.b = [];
        for (let l = 0; l < this.hlayer; l++) {
            this.w[l] = matrix_1.Matrix.generate(this.unit(l), this.unit(l - 1));
            this.b[l] = matrix_1.Matrix.generate(1, this.unit(l));
        }
        if (opt.mode)
            this.mode = opt.mode;
        if (opt.rate)
            this.rate = opt.rate;
        if (opt.w)
            this.w = opt.w;
        if (opt.b)
            this.b = opt.b;
        if (opt.scale)
            this.scale = opt.scale;
    }
    unit(l) {
        let n = this.shape[l + 1];
        return Array.isArray(n) ? n[0] : n;
    }
    af(l) {
        let n = this.shape[l + 1];
        return Array.isArray(n) ? n[1] : undefined;
    }
    afn(x, rows, af) {
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
    afd(x, af) {
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
    toJSON() {
        return JSON.stringify({
            mode: this.mode,
            shape: this.shape,
            rate: this.rate,
            scale: this.scale ? this.scale.dataSync() : undefined,
            w: this.w.map(w => w.dataSync()),
            b: this.b.map(b => b.dataSync()),
        });
    }
    static fromJSON(json) {
        let tmp = JSON.parse(json);
        let w = tmp.w.map((w) => new matrix_1.Matrix(w));
        let b = tmp.b.map((b) => new matrix_1.Matrix(b));
        let scale = tmp.scale ? new matrix_1.Matrix(tmp.scale) : undefined;
        return new BPNet(tmp.shape, {
            mode: tmp.mode,
            rate: tmp.mode,
            w, b, scale
        });
    }
    forwardPropagation(xs) {
        let hy = [];
        for (let l = 0; l < this.hlayer; l++) {
            let lastHy = l === 0 ? xs : hy[l - 1];
            let af = this.af(l);
            let tmp = lastHy.multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j));
            hy[l] = tmp.atomicOperation((item, i) => this.afn(item, tmp.getRow(i), af));
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
        let hy = this.predictNet(xs);
        return hy[hy.length - 1];
    }
    predictNet(xs) {
        this.checkInput(xs);
        xs = this.scaled(xs);
        let hy = this.forwardPropagation(xs);
        return [xs, ...hy];
    }
    backPropagationMultiple(hy, xs, ys) {
        let m = ys.shape[0];
        let dws = this.w.map(w => w.zeroed());
        let dys = this.b.map(b => b.zeroed());
        for (let n = 0; n < m; n++) {
            let nhy = hy.map(item => new matrix_1.Matrix([item.getRow(n)]));
            let nxs = new matrix_1.Matrix([xs.getRow(n)]);
            let nys = new matrix_1.Matrix([ys.getRow(n)]);
            let { dw: ndw, dy: ndy } = this.backPropagation(nhy, nxs, nys);
            dws = dws.map((d, l) => d.addition(ndw[l]));
            dys = dys.map((d, l) => d.addition(ndy[l]));
        }
        let dw = dws.map(d => d.atomicOperation(item => item / m));
        let dy = dys.map(d => d.atomicOperation(item => item / m));
        return { dy, dw };
    }
    backPropagation(hy, xs, ys) {
        let dw = [], dy = [];
        for (let l = this.hlayer - 1; l >= 0; l--) {
            let lastHy = hy[l - 1] ? hy[l - 1] : xs;
            let af = this.af(l);
            if (l === this.hlayer - 1) {
                dy[l] = hy[l].atomicOperation((item, r, c) => (item - ys.get(r, c)) * this.afd(item, af));
            }
            else {
                dy[l] = dy[l + 1].multiply(this.w[l + 1]).atomicOperation((item, r, c) => item * this.afd(hy[l].get(r, c), af));
            }
            dw[l] = dy[l].T.multiply(lastHy);
        }
        return { dy, dw };
    }
    adjust(dy, dw) {
        this.w = this.w.map((w, l) => w.subtraction(dw[l].numberMultiply(this.rate)));
        this.b = this.b.map((b, l) => b.subtraction(dy[l].numberMultiply(this.rate)));
    }
    cost(hy, ys) {
        let m = ys.shape[0];
        let sub = hy.subtraction(ys).atomicOperation(item => (item ** 2) / 2).columnSum();
        let tmp = sub.getRow(0).map(v => v / m);
        return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    calcLoss(xs, ys) {
        this.checkSample(xs, ys);
        let lastHy = this.predict(xs);
        return this.cost(lastHy, ys);
    }
    async bgd(xs, ys, opt) {
        for (let ep = 0; ep < opt.epochs; ep++) {
            let hy = this.forwardPropagation(xs);
            let { dy, dw } = this.backPropagationMultiple(hy, xs, ys);
            this.adjust(dy, dw);
            if (opt.onEpoch) {
                opt.onEpoch(ep, this.cost(hy[hy.length - 1], ys));
                opt.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (opt.onTrainEnd && ep === opt.epochs - 1) {
                opt.onTrainEnd(this.cost(hy[hy.length - 1], ys));
            }
        }
    }
    async sgd(xs, ys, opt) {
        let m = ys.shape[0];
        for (let ep = 0; ep < opt.epochs; ep++) {
            let hys = null;
            for (let n = 0; n < m; n++) {
                let nxs = new matrix_1.Matrix([xs.getRow(n)]);
                let nys = new matrix_1.Matrix([ys.getRow(n)]);
                let hy = this.forwardPropagation(nxs);
                const { dy, dw } = this.backPropagation(hy, nxs, nys);
                this.adjust(dy, dw);
                hys = hys ? hys.connect(hy[hy.length - 1]) : hy[hy.length - 1];
            }
            if (opt.onEpoch) {
                opt.onEpoch(ep, this.cost(hys, ys));
                opt.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (opt.onTrainEnd && ep === opt.epochs - 1) {
                opt.onTrainEnd(this.cost(hys, ys));
            }
        }
    }
    async mbgd(xs, ys, opt) {
        let m = ys.shape[0];
        let batchSize = opt.batchSize;
        let batch = Math.ceil(m / batchSize);
        for (let ep = 0; ep < opt.epochs; ep++) {
            let { xs: xst, ys: yst } = common_1.upset(xs, ys);
            let eploss = 0;
            for (let b = 0; b < batch; b++) {
                let start = b * batchSize;
                let end = start + batchSize;
                end = end > m ? m : end;
                let size = end - start;
                let bxs = xst.slice(start, end);
                let bys = yst.slice(start, end);
                let hy = this.forwardPropagation(bxs);
                let lastHy = hy[hy.length - 1];
                const { dy, dw } = this.backPropagationMultiple(hy, bxs, bys);
                this.adjust(dy, dw);
                let bloss = this.cost(lastHy, bys);
                eploss += bloss;
                if (opt.onBatch)
                    opt.onBatch(b, size, bloss);
            }
            if (opt.onEpoch) {
                opt.onEpoch(ep, eploss / batch);
                opt.async && await new Promise(resolve => setTimeout(resolve));
            }
            if (opt.onTrainEnd && ep === opt.epochs - 1) {
                opt.onTrainEnd(eploss / batch);
            }
        }
    }
    checkInput(xs) {
        if (xs.shape[1] !== this.unit(-1)) {
            throw new Error(`Input matrix column number error, input shape -> ${this.unit(-1)}.`);
        }
    }
    checkOutput(ys) {
        if (ys.shape[1] !== this.unit(this.hlayer - 1)) {
            throw new Error(`Output matrix column number error, output shape -> ${this.unit(this.hlayer - 1)}.`);
        }
    }
    checkSample(xs, ys) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('The row number of input and output matrix is not uniform.');
        }
        this.checkInput(xs);
        this.checkOutput(ys);
    }
    fit(xs, ys, opt = {}) {
        let m = ys.shape[0];
        let nopt = { ...exports.defaultTrainingOptions(m), ...opt };
        this.checkSample(xs, ys);
        if (nopt.batchSize > m) {
            throw new Error(`The batch size cannot be greater than the number of samples.`);
        }
        const [nxs, scale] = xs.normalization();
        this.scale = scale;
        xs = nxs;
        switch (this.mode) {
            case 'bgd':
                return this.bgd(xs, ys, nopt);
            case 'mbgd':
                return this.mbgd(xs, ys, nopt);
            case 'sgd':
            default:
                return this.sgd(xs, ys, nopt);
        }
    }
}
exports.BPNet = BPNet;
