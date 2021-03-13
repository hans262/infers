"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LogisticModel = exports.RegressionModel = void 0;
const matrix_1 = require("./matrix");
class RegressionModel {
    constructor(inputs, outputs) {
        this.rate = 0.01;
        const [inp, scalem] = inputs.normalization();
        this.scalem = scalem;
        this.inputs = inp.expansion(1);
        this.outputs = outputs;
        this.M = this.inputs.shape[0];
        this.weights = this.initWeights();
    }
    setRate(rate) {
        this.rate = rate;
    }
    initWeights() {
        const F = this.inputs.shape[1];
        return matrix_1.Matrix.generate(F, 1, 0);
    }
    hypothetical(xs) {
        return xs.multiply(this.weights);
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        let sum = 0;
        for (let i = 0; i < h.shape[0]; i++) {
            sum += (h.get(i, 0) - this.outputs.get(i, 0)) ** 2;
        }
        return (1 / (2 * M)) * sum;
    }
    gradientDescent() {
        const M = this.M;
        const F = this.weights.shape[0];
        const temps = this.initWeights();
        for (let i = 0; i < F; i++) {
            let h = this.hypothetical(this.inputs);
            let sum = 0;
            for (let j = 0; j < h.shape[0]; j++) {
                sum += (h.get(j, 0) - this.outputs.get(j, 0)) * this.inputs.get(j, i);
            }
            temps.update(i, 0, this.weights.get(i, 0) - this.rate * (1 / M) * sum);
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
                m.push((xs.get(i, j) - this.scalem.get(0, j)) / this.scalem.get(1, j));
            }
            n.push(m);
        }
        return new matrix_1.Matrix(n);
    }
    predict(xs) {
        let a = this.reductionScale(xs);
        return this.hypothetical(a.expansion(1));
    }
}
exports.RegressionModel = RegressionModel;
class LogisticModel {
    constructor(xs, ys) {
        this.rate = 0.01;
        this.inputs = xs.expansion(1);
        this.outputs = ys;
        this.weights = this.initWeights();
        this.M = this.inputs.shape[0];
    }
    setRate(rate) {
        this.rate = rate;
    }
    initWeights() {
        const F = this.inputs.shape[1];
        return matrix_1.Matrix.generate(F, 1, 0);
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        let sum = 0;
        for (let i = 0; i < h.shape[0]; i++) {
            let y = this.outputs.get(i, 0);
            let hy = h.get(i, 0);
            if (y === 1 && hy !== 0) {
                sum += -Math.log(hy);
            }
            if (y === 0 && hy !== 1) {
                sum += -Math.log(1 - hy);
            }
        }
        return (1 / M) * sum;
    }
    gradientDescent() {
        const M = this.M;
        const F = this.weights.shape[0];
        const temps = this.initWeights();
        for (let i = 0; i < F; i++) {
            let h = this.hypothetical(this.inputs);
            let sum = 0;
            for (let j = 0; j < h.shape[0]; j++) {
                sum += (h.get(j, 0) - this.outputs.get(j, 0)) * this.inputs.get(j, i);
            }
            temps.update(i, 0, this.weights.get(i, 0) - this.rate * (1 / M) * sum);
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
    predict(xs) {
        return this.hypothetical(xs.expansion(1));
    }
}
exports.LogisticModel = LogisticModel;
//# sourceMappingURL=model.js.map