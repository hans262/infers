export declare type MatrixShape = [number, number];
export declare class Matrix {
    shape: number[];
    self: number[][];
    constructor(data: number[][]);
    get(i: number): number[];
    det(): number;
    cominor(rowi: number, coli: number): Matrix;
    coLocationOperation(b: Matrix, oper: 'add' | 'sub'): Matrix;
    subtraction(b: Matrix): Matrix;
    addition(b: Matrix): Matrix;
    numberMultiply(b: number): Matrix;
    multiply(b: Matrix): Matrix;
    private scale;
    transposition(): Matrix;
    normalization(): Matrix;
    print(): void;
}
