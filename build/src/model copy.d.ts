import { Matrix } from "./matrix";
export declare class Model {
    weights: number[][];
    inputs: Matrix;
    outputs: Matrix;
    normal: Matrix;
    M: number;
    rate: number;
    constructor(inputs: Matrix, outputs: Matrix);
    setRate(rate: number): void;
    initWeights(): number[][];
    hypothetical(xs: Matrix): Matrix;
    cost(): number;
    gradientDescent(): void;
    fit(batch: number, callback?: (batch: number) => void): void;
    predict(xs: Matrix): Matrix;
}
