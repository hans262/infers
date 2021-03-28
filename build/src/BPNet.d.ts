import { Matrix } from "./matrix";
export declare type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh';
export declare type NetShape = (number | [number, ActivationFunction])[];
export interface NetConfig {
    optimizer: 'SGD' | 'BGD';
}
export declare class BPNet {
    readonly shape: NetShape;
    netconf?: NetConfig | undefined;
    w: Matrix[];
    b: Matrix[];
    nlayer: number;
    rate: number;
    scalem?: Matrix;
    constructor(shape: NetShape, netconf?: NetConfig | undefined);
    nOfLayer(l: number): number;
    afOfLayer(l: number): ActivationFunction | undefined;
    initwb(v?: number): Matrix[][];
    setRate(rate: number): void;
    afn(x: number, l: number): number;
    afd(x: number, l: number): number;
    calcnet(xs: Matrix): Matrix[];
    zoomScalem(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix[];
    calcDerivative(hy: Matrix[], ys: Matrix, n: number): {
        dy: Matrix[];
        dw: Matrix[];
    };
    update(dy: Matrix[], dw: Matrix[]): void;
    cost(hy: Matrix[], ys: Matrix): number;
    bgd(hy: Matrix[], ys: Matrix): void;
    sgd(hy: Matrix[], ys: Matrix): void;
    fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void): void;
}
