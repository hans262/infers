"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Matrix = void 0;
class Matrix {
    constructor(data) {
        this.shape = [1, 1];
        this.self = [[0]];
        let m = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length);
        if (m)
            throw new Error('矩阵列不正确');
        this.shape[0] = data.length;
        this.shape[1] = data[0].length;
        this.self = data;
    }
    static generate(row, col, f) {
        let n = [];
        for (let i = 0; i < row; i++) {
            let m = [];
            for (let j = 0; j < col; j++) {
                m.push(f);
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    update(row, col, val) {
        this.self[row][col] = val;
    }
    expansion(n, position) {
        let m = [];
        for (let i = 0; i < this.shape[0]; i++) {
            if (position === 'L') {
                m.push([n, ...this.getLine(i)]);
            }
            else {
                m.push([...this.getLine(i), n]);
            }
        }
        return new Matrix(m);
    }
    get(i, j) {
        return this.self[i][j];
    }
    getLine(i) {
        return this.self[i];
    }
    det() {
        if (this.shape[0] !== this.shape[1]) {
            throw new Error('只有方阵才能计算行列式');
        }
        if (this.shape[0] === 2 && this.shape[1] === 2) {
            return this.self[0][0] * this.self[1][1] - this.self[0][1] * this.self[1][0];
        }
        else {
            let m = 0;
            for (let i = 0; i < this.shape[1]; i++) {
                if (this.get(0, i) !== 0) {
                    m += this.get(0, i) * (-1) ** (i + 2) * this.cominor(1, i + 1).det();
                }
            }
            return m;
        }
    }
    cominor(rowi, coli) {
        if (this.shape[0] < 2 || this.shape[1] < 2) {
            throw new Error('求余子式行和列必须大于2才有意义');
        }
        let n = this.self.map((v) => {
            v = v.filter((_, j) => j !== coli - 1);
            return v;
        }).filter((_, i) => i !== rowi - 1);
        return new Matrix(n);
    }
    coLocationOperation(b, oper) {
        if (this.shape[0] !== b.shape[0] || this.shape[1] !== b.shape[1]) {
            throw new Error('必须满足两个矩阵是同形矩阵');
        }
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < this.shape[1]; j++) {
                let c = oper === 'add' ? this.self[i][j] + b.self[i][j] : this.self[i][j] - b.self[i][j];
                m.push(c);
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    subtraction(b) {
        return this.coLocationOperation(b, 'sub');
    }
    addition(b) {
        return this.coLocationOperation(b, 'add');
    }
    numberMultiply(b) {
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < this.shape[1]; j++) {
                m.push(this.self[i][j] * b);
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    multiply(b) {
        if (this.shape[1] !== b.shape[0]) {
            throw new Error('当矩阵A的列数等于矩阵B的行数，A与B才可以相乘');
        }
        let row = this.shape[0];
        let col = b.shape[1];
        let bt = b.transposition();
        let n = [];
        for (let i = 0; i < row; i++) {
            let m = [];
            for (let k = 0; k < col; k++) {
                let tm = this.self[i].reduce((p, c, j) => {
                    return p + c * bt.self[k][j];
                }, 0);
                m.push(tm);
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    scale() { }
    transposition() {
        let a = [];
        for (let i = 0; i < this.shape[1]; i++) {
            let n = [];
            for (let j = 0; j < this.shape[0]; j++) {
                n.push(this.self[j][i]);
            }
            a.push(n);
        }
        return new Matrix(a);
    }
    normalization() {
        let t = this.transposition();
        let n = [];
        for (let i = 0; i < t.shape[0]; i++) {
            const max = Math.max(...t.self[i]);
            const min = Math.min(...t.self[i]);
            const range = max - min;
            const average = min + (range / 2);
            n.push([average, range]);
            for (let j = 0; j < t.shape[1]; j++) {
                t.self[i][j] = range === 0 ? 0 : (t.self[i][j] - average) / range;
            }
        }
        return [t.transposition(), new Matrix(n).transposition()];
    }
    print() {
        console.log(`Matrix ${this.shape[0]}x${this.shape[1]} [`);
        for (let i = 0; i < this.shape[0]; i++) {
            let line = ' ';
            for (let j = 0; j < this.shape[1]; j++) {
                line += this.self[i][j] + ', ';
            }
            console.log(line);
        }
        console.log(']');
    }
}
exports.Matrix = Matrix;
//# sourceMappingURL=matrix.js.map