import { Matrix } from "./matrix";
export declare class NeuralNetwork {
    w: number[][][];
    b: number[][];
    shape: number[];
    layerNum: number;
    rate: number;
    constructor(shape: number[]);
    sigmoid(x: number): number;
    predict(xs: number[]): number[][];
    calcdelta(ys: number[], hy: number[][]): number[][];
    update(hy: number[][], delta: number[][]): void;
    fit(xs: Matrix, ys: Matrix, batch: number): void;
}
