"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LogisticModel = exports.RegressionModel = void 0;
const matrix_1 = require("./matrix");
class Model {
    constructor(xs, ys) {
        this.rate = 0.01;
        if (xs.shape[0] !== ys.shape[0]) {
            throw new Error('输入输出矩阵行数不统一');
        }
        const [inputs, scalem] = xs.normalization();
        this.scalem = scalem;
        this.inputs = inputs.expand(1, 'L');
        this.outputs = ys;
        this.M = this.inputs.shape[0];
        this.weights = this.initWeights();
    }
    setRate(rate) {
        this.rate = rate;
    }
    initWeights() {
        const F = this.inputs.shape[1];
        const y = this.outputs.shape[1];
        return matrix_1.Matrix.generate(F, y);
    }
    hypothetical(xs) {
        return xs.multiply(this.weights);
    }
    cost() {
        const M = this.M;
        let hy = this.hypothetical(this.inputs);
        let sub = hy.subtraction(this.outputs).atomicOperation(item => item ** 2);
        let n = [];
        for (let i = 0; i < sub.shape[1]; i++) {
            let sum = sub.getCol(i).reduce((p, c) => p + c);
            n.push((1 / (2 * M)) * sum);
        }
        return n;
    }
    gradientDescent() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        const temps = this.initWeights();
        let hsub = h.subtraction(this.outputs);
        for (let i = 0; i < temps.shape[0]; i++) {
            for (let j = 0; j < temps.shape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < hsub.shape[0]; k++) {
                    sum += hsub.get(k, j) * this.inputs.get(k, i);
                }
                let nw = this.weights.get(i, j) - this.rate * (1 / M) * sum;
                temps.update(i, j, nw);
            }
        }
        this.weights = temps;
    }
    fit(batch, callback) {
        for (let i = 0; i < batch; i++) {
            this.gradientDescent();
            if (callback) {
                callback(i);
            }
        }
    }
    zoomScale(xs) {
        let n = [];
        for (let i = 0; i < xs.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < xs.shape[1]; j++) {
                let r = this.scalem.get(1, j) === 0 ? 0 : (xs.get(i, j) - this.scalem.get(0, j)) / this.scalem.get(1, j);
                m.push(r);
            }
            n.push(m);
        }
        return new matrix_1.Matrix(n);
    }
    predict(xs) {
        if (xs.shape[1] !== this.inputs.shape[1] - 1) {
            throw new Error('与预期特征数不符合');
        }
        let inputs = this.zoomScale(xs).expand(1, 'L');
        return this.hypothetical(inputs);
    }
}
class RegressionModel extends Model {
}
exports.RegressionModel = RegressionModel;
class LogisticModel extends Model {
    constructor(xs, ys) {
        super(xs, ys);
        this.verifOutput(ys);
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
    cost() {
        const M = this.M;
        let hy = this.hypothetical(this.inputs);
        let t = hy.atomicOperation((item, i, j) => {
            let y = this.outputs.get(i, j);
            return y === 1 ? -Math.log(item) : -Math.log(1 - item);
        });
        let n = [];
        for (let i = 0; i < t.shape[1]; i++) {
            let sum = t.getCol(i).reduce((p, c) => p + c);
            n.push((1 / M) * sum);
        }
        return n;
    }
    sigmoid(x) {
        return 1 / (1 + Math.E ** -x);
    }
    hypothetical(xs) {
        let a = xs.multiply(this.weights);
        return a.atomicOperation(item => this.sigmoid(item));
    }
}
exports.LogisticModel = LogisticModel;
//# sourceMappingURL=model.js.map