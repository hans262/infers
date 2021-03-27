import { Matrix } from "./matrix";
export declare type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh';
export declare type BPNetShape = (number | [number, ActivationFunction])[];
export declare class BPNet {
    readonly shape: BPNetShape;
    w: Matrix[];
    b: Matrix[];
    nlayer: number;
    rate: number;
    scalem?: Matrix;
    constructor(shape: BPNetShape);
    nOfLayer(l: number): number;
    afOfLayer(l: number): ActivationFunction | undefined;
    initwb(v?: number): Matrix[][];
    setRate(rate: number): void;
    afn(x: number, l: number): number;
    afd(x: number, l: number): number;
    calcnet(xs: Matrix): Matrix[];
    zoomScalem(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix[];
    calcDerivative(ys: Matrix, hy: Matrix[]): {
        dy: Matrix[];
        dw: Matrix[];
    };
    update(dy: Matrix[], dw: Matrix[]): void;
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
}
