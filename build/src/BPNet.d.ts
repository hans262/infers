import { Matrix } from "./matrix";
export declare type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh';
export declare class BPNet {
    readonly shape: number[];
    readonly af?: "Sigmoid" | "Relu" | "Tanh" | undefined;
    w: Matrix[];
    b: Matrix[];
    nlayer: number;
    rate: number;
    scalem?: Matrix;
    constructor(shape: number[], af?: "Sigmoid" | "Relu" | "Tanh" | undefined);
    initwb(shape: number[]): Matrix[][];
    setRate(rate: number): void;
    afn(x: number): number;
    afd(x: number): number;
    calcnet(xs: Matrix): Matrix[];
    zoomScalem(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix[];
    calcDerivative(ys: number[], hy: Matrix[]): number[][];
    update(dy: number[][], hy: Matrix[]): void;
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
}
