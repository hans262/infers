import { Matrix } from "./matrix";
import { BPNet } from "./BPNet";
export declare function toFixed(num: number, fix: number): number;
export declare function upset(xs: Matrix, ys: Matrix): {
    xs: Matrix;
    ys: Matrix;
};
export declare function loadBPNet(modelJson: string): BPNet;
