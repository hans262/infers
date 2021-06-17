export interface GenerateMatrixOptions {
    range: [number, number];
    integer?: boolean;
}
export declare class Matrix {
    shape: [number, number];
    private self;
    constructor(data: number[][]);
    slice(start: number, end: number): Matrix;
    argMax(row: number): number;
    connect(b: Matrix): Matrix;
    zeroed(): Matrix;
    clone(): Matrix;
    getMeanOfRow(i: number): number;
    sum(): number;
    columnSum(): Matrix;
    dataSync(): number[][];
    equalsShape(b: Matrix): boolean;
    equals(b: Matrix): boolean;
    static generate(row: number, col: number, opt?: GenerateMatrixOptions | number): Matrix;
    update(row: number, col: number, val: number, oper?: '+=' | '-=' | '*=' | '/='): void;
    expand(n: number, position: 'L' | 'R' | 'T' | 'B'): Matrix;
    get(i: number, j: number): number;
    getRow(i: number): number[];
    getCol(k: number): number[];
    det(): number;
    cominor(rowi: number, coli: number): Matrix;
    atomicOperation(callback: (item: number, row: number, col: number) => number): Matrix;
    coLocationOperation(b: Matrix, oper: 'add' | 'sub' | 'mul' | 'exp'): Matrix;
    subtraction(b: Matrix): Matrix;
    addition(b: Matrix): Matrix;
    numberMultiply(b: number): Matrix;
    multiply(b: Matrix): Matrix;
    get T(): Matrix;
    normalization(): Matrix[];
    print(): void;
}
