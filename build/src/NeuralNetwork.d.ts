import { Matrix } from "./matrix";
export declare class NeuralNetwork {
    w: number[][][];
    b: number[][];
    readonly shape: number[];
    layerNum: number;
    rate: number;
    esp: number;
    constructor(shape: number[]);
    setRate(rate: number): void;
    sigmoid(x: number): number;
    predict(xs: Matrix): Matrix[];
    calcNetwork(xs: number[]): number[][];
    calcdelta(ys: number[], hy: number[][]): number[][];
    update(hy: number[][], delta: number[][]): void;
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
}
