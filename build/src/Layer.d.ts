import { ActivationFunction } from "./BPNet";
import { Matrix } from "./matrix";
export declare class Layer {
    index: number;
    unit: number;
    lastUnit: number;
    lastLayer?: Layer | undefined;
    af?: ActivationFunction | undefined;
    w: Matrix;
    b: Matrix;
    hy?: Matrix;
    constructor(index: number, unit: number, lastUnit: number, lastLayer?: Layer | undefined, af?: ActivationFunction | undefined);
    calchy(xs?: Matrix): void;
    afn(x: number): number;
    afd(x: number): number;
}
