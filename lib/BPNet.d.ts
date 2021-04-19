import { Matrix } from "./matrix";
import { ActivationFunction, FitConf, Mode, NetConfig, NetShape } from "./types";
export declare class BPNet {
    readonly shape: NetShape;
    w: Matrix[];
    b: Matrix[];
    hlayer: number;
    scale?: Matrix;
    mode: Mode;
    rate: number;
    constructor(shape: NetShape, conf?: NetConfig);
    unit(l: number): number;
    af(l: number): ActivationFunction | undefined;
    afn(x: number, rows: number[], af?: ActivationFunction): number;
    afd(x: number, af?: ActivationFunction): number;
    toJSON(): string;
    calcnet(xs: Matrix): Matrix[];
    scaled(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
    calcDerivativeMul(hy: Matrix[], xs: Matrix, ys: Matrix): {
        dy: Matrix[];
        dw: Matrix[];
    };
    calcDerivative(hy: Matrix[], xs: Matrix, ys: Matrix): {
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
