import { Matrix } from "./matrix";
export declare class SeqModel {
    readonly shape: number[];
    readonly af?: "Sigmoid" | undefined;
    weights: Matrix;
    scalem?: Matrix;
    rate: number;
    constructor(shape: number[], af?: "Sigmoid" | undefined);
    setRate(rate: number): void;
    initw(): Matrix;
    hypothetical(xs: Matrix): Matrix;
    cost(hys: Matrix, ys: Matrix): number[];
    sigmoidCost(hys: Matrix, ys: Matrix): number[];
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
    sigmoid(x: number): number;
    verifOutput(ys: Matrix): void;
    zoomScalem(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
}
