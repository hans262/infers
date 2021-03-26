"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.SeqModel = void 0;
const matrix_1 = require("./matrix");
class SeqModel {
    constructor(shape, af) {
        this.shape = shape;
        this.af = af;
        this.rate = 0.01;
        if (shape.length !== 2) {
            throw new Error('该模型只支持两层结构');
        }
        this.weights = this.initw();
    }
    setRate(rate) {
        this.rate = rate;
    }
    initw() {
        const f = this.shape[0] + 1;
        const y = this.shape[1];
        return matrix_1.Matrix.generate(f, y);
    }
    hypothetical(xs) {
        let a = xs.multiply(this.weights);
        return this.af === 'Sigmoid' ? a.atomicOperation(i => this.sigmoid(i)) : a;
    }
    cost(hys, ys) {
        let m = hys.shape[0];
        let sub = hys.subtraction(ys).atomicOperation(item => item ** 2).columnSum();
        return sub.getRow(0).map(v => (1 / (2 * m)) * v);
    }
    sigmoidCost(hys, ys) {
        let m = hys.shape[0];
        let t = hys.atomicOperation((hy, i, j) => {
            let y = ys.get(i, j);
            return y === 1 ? -Math.log(hy) : -Math.log(1 - hy);
        }).columnSum();
        return t.getRow(0).map(v => (1 / m) * v);
    }
    fit(xs, ys, batch, callback) {
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('输入输出矩阵行数不统一');
        }
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`);
        }
        if (ys.shape[1] !== this.shape[1]) {
            throw new Error(`标签与网络输出不符合，output num -> ${this.shape[1]}`);
        }
        if (this.af === 'Sigmoid') {
            this.verifOutput(ys);
        }
        let [inputs, scalem] = xs.normalization();
        this.scalem = scalem;
        xs = inputs.expand(1, 'L');
        let m = xs.shape[0];
        for (let i = 0; i < batch; i++) {
            let hys = this.hypothetical(xs);
            const temps = this.initw();
            let hsub = hys.subtraction(ys);
            for (let i = 0; i < temps.shape[0]; i++) {
                for (let j = 0; j < temps.shape[1]; j++) {
                    let sum = 0;
                    for (let k = 0; k < hsub.shape[0]; k++) {
                        sum += hsub.get(k, j) * xs.get(k, i);
                    }
                    let nw = this.weights.get(i, j) - this.rate * (1 / m) * sum;
                    temps.update(i, j, nw);
                }
            }
            this.weights = temps;
            let loss = this.af === 'Sigmoid' ? this.sigmoidCost(hys, ys)[0] : this.cost(hys, ys)[0];
            if (callback)
                callback(i, loss);
        }
    }
    sigmoid(x) {
        return 1 / (1 + Math.E ** -x);
    }
    verifOutput(ys) {
        for (let i = 0; i < ys.shape[0]; i++) {
            if (ys.shape[1] > 1 && ys.getRow(i).reduce((p, c) => p + c) !== 1)
                throw new Error('输出矩阵每行求和必须等0');
            for (let j = 0; j < ys.shape[1]; j++) {
                if (ys.get(i, j) !== 0 && ys.get(i, j) !== 1)
                    throw new Error('输出矩阵属于域 ∈ (0, 1)');
            }
        }
    }
    zoomScalem(xs) {
        return xs.atomicOperation((item, _, j) => {
            if (!this.scalem)
                return item;
            return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j);
        });
    }
    predict(xs) {
        if (xs.shape[1] !== this.shape[0]) {
            throw new Error('与预期特征数不符合');
        }
        let inputs = this.zoomScalem(xs).expand(1, 'L');
        return this.hypothetical(inputs);
    }
}
exports.SeqModel = SeqModel;
//# sourceMappingURL=SeqModel.js.map