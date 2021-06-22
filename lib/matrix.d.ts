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
    getMeanOfRow(row: number): number;
    sum(): number;
    columnSum(): Matrix;
    dataSync(): number[][];
    equalsShape(b: Matrix): boolean;
    equals(b: Matrix): boolean;
    static generate(row: number, col: number, opt?: GenerateMatrixOptions | number): Matrix;
    update(row: number, col: number, val: number, oper?: '+=' | '-=' | '*=' | '/='): void;
    expand(n: number, position: 'L' | 'R' | 'T' | 'B'): Matrix;
    get(row: number, col: number): number;
    getRow(row: number): number[];
    getCol(col: number): number[];
    adjugate(): Matrix;
    inverse(): Matrix;
    det(): number;
    cominor(row: number, col: number): Matrix;
    atomicOperation(callback: (item: number, row: number, col: number) => number): Matrix;
    coLocationOperation(b: Matrix, oper: 'add' | 'sub' | 'mul' | 'exp'): Matrix;
    subtraction(b: Matrix): Matrix;
    addition(b: Matrix): Matrix;
    multiply(b: Matrix | number): Matrix;
    get T(): Matrix;
    normalization(): Matrix[];
    print(): void;
}
