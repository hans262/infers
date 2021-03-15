export declare class Matrix {
    shape: [number, number];
    private self;
    constructor(data: number[][]);
    dataSync(): number[][];
    equalsShape(b: Matrix): boolean;
    equals(b: Matrix): boolean;
    static generate(row: number, col: number, f?: number): Matrix;
    update(row: number, col: number, val: number): void;
    expand(n: number, position: 'L' | 'R'): Matrix;
    get(i: number, j: number): number;
    getRow(i: number): number[];
    getCol(k: number): number[];
    det(): number;
    cominor(rowi: number, coli: number): Matrix;
    atomicOperation(callback: (item: number, row: number, col: number) => number): Matrix;
    coLocationOperation(b: Matrix, oper: 'add' | 'sub'): Matrix;
    subtraction(b: Matrix): Matrix;
    addition(b: Matrix): Matrix;
    numberMultiply(b: number): Matrix;
    multiply(b: Matrix): Matrix;
    get T(): Matrix;
    normalization(): Matrix[];
    print(): void;
}
