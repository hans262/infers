import { Matrix } from "./matrix";
import { FitConf, Mode, NetConfig, NetShape } from "./types";
export declare class BPNet {
    readonly shape: NetShape;
    w: Matrix[];
    b: Matrix[];
    nlayer: number;
    scale?: Matrix;
    mode: Mode;
    rate: number;
    constructor(shape: NetShape, conf?: NetConfig);
    unit(l: number): number;
    af(l: number): import("./types").ActivationFunction | undefined;
    afn(x: number, l: number, rows: number[]): number;
    afd(x: number, l: number): number;
    calcnet(xs: Matrix): Matrix[];
    scaled(xs: Matrix): Matrix;
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
    bgd(xs: Matrix, ys: Matrix, conf: FitConf): Promise<void>;
    sgd(xs: Matrix, ys: Matrix, conf: FitConf): Promise<void>;
    mbgd(xs: Matrix, ys: Matrix, conf: FitConf): Promise<void>;
    fit(xs: Matrix, ys: Matrix, conf: FitConf): Promise<void>;
}
