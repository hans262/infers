import { Matrix } from "./matrix";
declare type ActivationFunction = 'Sigmoid' | 'Relu';
export declare class BPNet {
    readonly shape: number[];
    readonly af?: ActivationFunction | undefined;
    w: Matrix[];
    b: Matrix[];
    layerNum: number;
    rate: number;
    constructor(shape: number[], af?: ActivationFunction | undefined);
    initwb(shape: number[]): Matrix[][];
    setRate(rate: number): void;
    afn(x: number): number;
    afd(x: number): number;
    predict(xs: Matrix): Matrix[];
    calcNetwork(xs: number[]): number[][];
    calcdelta(ys: number[], hy: number[][]): number[][];
    update(hy: number[][], delta: number[][]): void;
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
}
export {};
