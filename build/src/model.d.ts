import { Matrix } from "./matrix";
export declare class Model {
    weights: Matrix;
    inputs: Matrix;
    outputs: Matrix;
    M: number;
    rate: number;
    constructor(inputs: Matrix, outputs: Matrix);
    setRate(rate: number): void;
    initWeights(): Matrix;
    hypothetical(xs: Matrix): Matrix;
    cost(): number;
    gradientDescent(): void;
    fit(batch: number, callback?: (batch: number) => void): void;
    predict(xs: Matrix): Matrix;
}
