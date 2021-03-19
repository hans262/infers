"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
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
        this.m = this.inputs.shape[0];
        this.weights = this.initWeights();
    }
    setRate(rate) {
        this.rate = rate;
    }
    initWeights() {
        const f = this.inputs.shape[1];
        const y = this.outputs.shape[1];
        return matrix_1.Matrix.generate(f, y);
    }
    hypothetical(xs) {
        return xs.multiply(this.weights);
    }
    cost() {
        let h = this.hypothetical(this.inputs);
        let sub = h.subtraction(this.outputs).atomicOperation(item => item ** 2).columnSum();
        return sub.getRow(0).map(v => (1 / (2 * this.m)) * v);
    }
    gradientDescent() {
        let h = this.hypothetical(this.inputs);
        const temps = this.initWeights();
        let hsub = h.subtraction(this.outputs);
        for (let i = 0; i < temps.shape[0]; i++) {
            for (let j = 0; j < temps.shape[1]; j++) {
                let sum = 0;
                for (let k = 0; k < hsub.shape[0]; k++) {
                    sum += hsub.get(k, j) * this.inputs.get(k, i);
                }
                let nw = this.weights.get(i, j) - this.rate * (1 / this.m) * sum;
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
        return xs.atomicOperation((item, _, j) => {
            return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j);
        });
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
        let h = this.hypothetical(this.inputs);
        let t = h.atomicOperation((hy, i, j) => {
            let y = this.outputs.get(i, j);
            return y === 1 ? -Math.log(hy) : -Math.log(1 - hy);
        }).columnSum();
        return t.getRow(0).map(v => (1 / this.m) * v);
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