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
        const [inp, scalem] = xs.normalization();
        this.scalem = scalem;
        this.inputs = inp.expansion(1, 'L');
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
        return matrix_1.Matrix.generate(F, y, 0);
    }
    hypothetical(xs) {
        return xs.multiply(this.weights);
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        let n = [];
        for (let i = 0; i < h.shape[1]; i++) {
            let sum = 0;
            for (let j = 0; j < h.shape[0]; j++) {
                sum += (h.get(j, i) - this.outputs.get(j, i)) ** 2;
            }
            n.push((1 / (2 * M)) * sum);
        }
        return n;
    }
    gradientDescent() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        const temps = this.initWeights();
        for (let i = 0; i < temps.shape[0]; i++) {
            for (let j = 0; j < temps.shape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < h.shape[0]; k++) {
                    sum += (h.get(k, j) - this.outputs.get(k, j)) * this.inputs.get(k, i);
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
    reductionScale(xs) {
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
        let a = this.reductionScale(xs);
        return this.hypothetical(a.expansion(1, 'L'));
    }
}
class RegressionModel extends Model {
}
exports.RegressionModel = RegressionModel;
class LogisticModel extends Model {
    constructor(xs, ys) {
        super(xs, ys);
        this.verifYs(ys);
    }
    verifYs(ys) {
        for (let i = 0; i < ys.shape[0]; i++) {
            if (ys.shape[1] > 1 && ys.getLine(i).reduce((p, c) => p + c) !== 1)
                throw new Error('输出矩阵每行和必须等0');
            for (let j = 0; j < ys.shape[1]; j++) {
                if (ys.get(i, j) !== 0 && ys.get(i, j) !== 1)
                    throw new Error('输出矩阵属于域 ∈ (0, 1)');
            }
        }
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        let n = [];
        for (let j = 0; j < h.shape[1]; j++) {
            let sum = 0;
            for (let i = 0; i < h.shape[0]; i++) {
                let y = this.outputs.get(i, 0);
                let hy = h.get(i, 0);
                if (y === 1) {
                    sum += -Math.log(hy);
                }
                if (y === 0) {
                    sum += -Math.log(1 - hy);
                }
            }
            n.push((1 / M) * sum);
        }
        return n;
    }
    sigmoid(x) {
        return 1 / (1 + Math.E ** -x);
    }
    hypothetical(xs) {
        let a = xs.multiply(this.weights);
        let n = [];
        for (let i = 0; i < a.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < a.shape[1]; j++) {
                m.push(this.sigmoid(a.get(i, j)));
            }
            n.push(m);
        }
        return new matrix_1.Matrix(n);
    }
}
exports.LogisticModel = LogisticModel;
//# sourceMappingURL=model.js.map