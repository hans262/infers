"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Model = void 0;
const matrix_1 = require("./matrix");
class Model {
    constructor(inputs, outputs) {
        this.rate = 0.01;
        this.inputs = inputs.expansion(1);
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
    predict(xs) {
        return this.hypothetical(xs.expansion(1));
    }
}
exports.Model = Model;
//# sourceMappingURL=model.js.map