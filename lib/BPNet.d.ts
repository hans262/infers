import { Matrix } from "./matrix";
import type { ActivationFunction, TrainingOptions, Mode, BPNetOptions, NetShape } from "./types";
export declare const defaultTrainingOptions: (m: number) => TrainingOptions;
export declare class BPNet {
    readonly shape: NetShape;
    w: Matrix[];
    b: Matrix[];
    hlayer: number;
    scale?: Matrix;
    mode: Mode;
    rate: number;
    constructor(shape: NetShape, opt?: BPNetOptions);
    unit(l: number): number;
    af(l: number): ActivationFunction | undefined;
    afn(x: number, rows: number[], af?: ActivationFunction): number;
    afd(x: number, af?: ActivationFunction): number;
    toJSON(): string;
    static fromJSON(json: string): BPNet;
    forwardPropagation(xs: Matrix): Matrix[];
    scaled(xs: Matrix): Matrix;
    predict(xs: Matrix): Matrix;
    predictNet(xs: Matrix): Matrix[];
    backPropagationMultiple(hy: Matrix[], xs: Matrix, ys: Matrix): {
        dy: Matrix[];
        dw: Matrix[];
    };
    backPropagation(hy: Matrix[], xs: Matrix, ys: Matrix): {
        dy: Matrix[];
        dw: Matrix[];
    };
    adjust(dy: Matrix[], dw: Matrix[]): void;
    cost(hy: Matrix, ys: Matrix): number;
    calcLoss(xs: Matrix, ys: Matrix): number;
    bgd(xs: Matrix, ys: Matrix, opt: TrainingOptions): Promise<void>;
    sgd(xs: Matrix, ys: Matrix, opt: TrainingOptions): Promise<void>;
    mbgd(xs: Matrix, ys: Matrix, opt: TrainingOptions): Promise<void>;
    checkInput(xs: Matrix): void;
    checkOutput(ys: Matrix): void;
    checkSample(xs: Matrix, ys: Matrix): void;
    fit(xs: Matrix, ys: Matrix, opt?: Partial<TrainingOptions>): Promise<void>;
}
