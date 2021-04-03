import { Matrix } from "./matrix";
export declare type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh' | 'Softmax';
export declare type Mode = 'sgd' | 'bgd' | 'mbgd';
export declare type NetShape = (number | [number, ActivationFunction])[];
export interface NetConfig {
    mode?: Mode;
    rate?: number;
    w?: Matrix[];
    b?: Matrix[];
    scalem?: Matrix;
}
export interface FitConf {
    epochs: number;
    batchSize?: number;
    onBatch?: (batch: number, size: number, loss: number) => void;
    onEpoch?: (epoch: number, loss: number) => void;
}
export declare class BPNet {
    readonly shape: NetShape;
    w: Matrix[];
    b: Matrix[];
    nlayer: number;
    scalem?: Matrix;
    mode: Mode;
    rate: number;
    constructor(shape: NetShape, conf?: NetConfig);
    nOfLayer(l: number): number;
    afOfLayer(l: number): ActivationFunction | undefined;
    initwb(v?: number): Matrix[][];
    afn(x: number, l: number, rows: number[]): number;
    afd(x: number, l: number): number;
    calcnet(xs: Matrix): Matrix[];
    zoomScalem(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
    calcDerivativeMul(hy: Matrix[], ys: Matrix): {
        dy: Matrix[];
        dw: Matrix[];
    };
    calcDerivative(hy: Matrix[], ys: Matrix): {
        dy: Matrix[];
        dw: Matrix[];
    };
    update(dy: Matrix[], dw: Matrix[]): void;
    cost(hy: Matrix, ys: Matrix): number;
    bgd(xs: Matrix, ys: Matrix, conf: FitConf): void;
    sgd(xs: Matrix, ys: Matrix, conf: FitConf): void;
    mbgd(xs: Matrix, ys: Matrix, conf: FitConf): void;
    upset(xs: Matrix, ys: Matrix): {
        xs: Matrix;
        ys: Matrix;
    };
    fit(xs: Matrix, ys: Matrix, conf: FitConf): void;
}
