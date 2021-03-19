import { Matrix } from "./matrix";
declare class Model {
    weights: Matrix;
    inputs: Matrix;
    outputs: Matrix;
    scalem: Matrix;
    m: number;
    rate: number;
    constructor(xs: Matrix, ys: Matrix);
    setRate(rate: number): void;
    initWeights(): Matrix;
    hypothetical(xs: Matrix): Matrix;
    cost(): number[];
    gradientDescent(): void;
    fit(batch: number, callback?: (batch: number) => void): void;
    zoomScale(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
}
export declare class RegressionModel extends Model {
}
export declare class LogisticModel extends Model {
    constructor(xs: Matrix, ys: Matrix);
    verifOutput(ys: Matrix): void;
    cost(): number[];
    sigmoid(x: number): number;
    hypothetical(xs: Matrix): Matrix;
}
export {};
