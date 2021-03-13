import { Matrix } from "./matrix";
export declare class RegressionModel {
    weights: Matrix;
    inputs: Matrix;
    outputs: Matrix;
    scalem: Matrix;
    M: number;
    rate: number;
    constructor(inputs: Matrix, outputs: Matrix);
    setRate(rate: number): void;
    initWeights(): Matrix;
    hypothetical(xs: Matrix): Matrix;
    cost(): number;
    gradientDescent(): void;
    fit(batch: number, callback?: (batch: number) => void): void;
    reductionScale(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
}
export declare class LogisticModel {
    weights: Matrix;
    inputs: Matrix;
    outputs: Matrix;
    rate: number;
    M: number;
    constructor(xs: Matrix, ys: Matrix);
    setRate(rate: number): void;
    initWeights(): Matrix;
    cost(): number;
    gradientDescent(): void;
    fit(batch: number, callback?: (batch: number) => void): void;
    sigmoid(x: number): number;
    hypothetical(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
}
