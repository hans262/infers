export declare type MatrixShape = [number, number];
export declare class Matrix {
    shape: number[];
    private self;
    constructor(data: number[][]);
    static generate(row: number, col: number, f: number): Matrix;
    update(row: number, col: number, val: number): void;
    expansion(n: number): Matrix;
    get(i: number, j: number): number;
    getLine(i: number): number[];
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
