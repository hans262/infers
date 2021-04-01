"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Matrix = void 0;
class Matrix {
    constructor(data) {
        let t = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length);
        if (t)
            throw new Error('矩阵列不正确');
        this.shape = [data.length, data[0].length];
        this.self = data;
    }
    zeroed() {
        return this.atomicOperation(_ => 0);
    }
    clone() {
        return new Matrix(this.dataSync());
    }
    getMeanOfRow(i) {
        let tmp = this.getRow(i);
        return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    columnSum() {
        let n = [];
        for (let i = 0; i < this.shape[1]; i++) {
            n.push(this.getCol(i).reduce((p, c) => p + c));
        }
        return new Matrix([n]);
    }
    dataSync() {
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < this.shape[1]; j++) {
                m.push(this.get(i, j));
            }
            n.push(m);
        }
        return n;
    }
    equalsShape(b) {
        return this.shape[0] === b.shape[0] && this.shape[1] === b.shape[1];
    }
    equals(b) {
        if (!this.equalsShape(b)) {
            return false;
        }
        for (let i = 0; i < this.shape[0]; i++) {
            for (let j = 0; j < this.shape[1]; j++) {
                if (this.get(i, j) !== b.get(i, j))
                    return false;
            }
        }
        return true;
    }
    static generate(row, col, f) {
        let n = [];
        for (let i = 0; i < row; i++) {
            let m = [];
            for (let j = 0; j < col; j++) {
                m.push(f ? f : 0.5 - Math.random());
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    update(row, col, val, oper) {
        switch (oper) {
            case '+=':
                this.self[row][col] += val;
                break;
            case '-=':
                this.self[row][col] -= val;
                break;
            case '*=':
                this.self[row][col] *= val;
                break;
            case '/=':
                this.self[row][col] /= val;
                break;
            default:
                this.self[row][col] = val;
        }
    }
    expand(n, position) {
        let m = [];
        for (let i = 0; i < this.shape[0]; i++) {
            if (position === 'L') {
                m.push([n, ...this.getRow(i)]);
            }
            else {
                m.push([...this.getRow(i), n]);
            }
        }
        return new Matrix(m);
    }
    get(i, j) {
        return this.self[i][j];
    }
    getRow(i) {
        return [...this.self[i]];
    }
    getCol(k) {
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            for (let j = 0; j < this.shape[1]; j++) {
                if (j === k) {
                    n.push(this.get(i, j));
                }
            }
        }
        return n;
    }
    det() {
        if (this.shape[0] !== this.shape[1]) {
            throw new Error('只有方阵才能计算行列式');
        }
        if (this.shape[0] === 2 && this.shape[1] === 2) {
            return this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0);
        }
        else {
            let m = 0;
            for (let i = 0; i < this.shape[1]; i++) {
                if (this.get(0, i) !== 0) {
                    m += this.get(0, i) * ((-1) ** (i + 2)) * this.cominor(0, i).det();
                }
            }
            return m;
        }
    }
    cominor(rowi, coli) {
        if (this.shape[0] < 2 || this.shape[1] < 2) {
            throw new Error('求余子式行和列必须大于2才有意义');
        }
        let n = this.dataSync().map((v) => {
            v = v.filter((_, j) => j !== coli);
            return v;
        }).filter((_, i) => i !== rowi);
        return new Matrix(n);
    }
    atomicOperation(callback) {
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < this.shape[1]; j++) {
                m.push(callback(this.get(i, j), i, j));
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    coLocationOperation(b, oper) {
        if (!this.equalsShape(b)) {
            throw new Error('必须满足两个矩阵是同形矩阵');
        }
        let n = [];
        for (let i = 0; i < this.shape[0]; i++) {
            let m = [];
            for (let j = 0; j < this.shape[1]; j++) {
                let c = oper === 'add' ? this.get(i, j) + b.get(i, j) : this.get(i, j) - b.get(i, j);
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
        return this.atomicOperation(item => item * b);
    }
    multiply(b) {
        if (this.shape[1] !== b.shape[0]) {
            throw new Error('当矩阵A的列数等于矩阵B的行数，A与B才可以相乘');
        }
        let row = this.shape[0];
        let col = b.shape[1];
        let bt = b.T;
        let n = [];
        for (let i = 0; i < row; i++) {
            let m = [];
            for (let k = 0; k < col; k++) {
                let tm = this.getRow(i).reduce((p, c, j) => {
                    return p + c * bt.get(k, j);
                }, 0);
                m.push(tm);
            }
            n.push(m);
        }
        return new Matrix(n);
    }
    get T() {
        let a = [];
        for (let i = 0; i < this.shape[1]; i++) {
            let n = [];
            for (let j = 0; j < this.shape[0]; j++) {
                n.push(this.get(j, i));
            }
            a.push(n);
        }
        return new Matrix(a);
    }
    normalization() {
        let t = this.T;
        let n = [];
        for (let i = 0; i < t.shape[0]; i++) {
            const max = Math.max(...t.getRow(i));
            const min = Math.min(...t.getRow(i));
            const range = max - min;
            const average = min + (range / 2);
            n.push([average, range]);
            for (let j = 0; j < t.shape[1]; j++) {
                let s = range === 0 ? 0 : (t.get(i, j) - average) / range;
                t.update(i, j, s);
            }
        }
        return [t.T, new Matrix(n).T];
    }
    print() {
        console.log(`Matrix ${this.shape[0]}x${this.shape[1]} [`);
        for (let i = 0; i < this.shape[0]; i++) {
            let line = ' ';
            for (let j = 0; j < this.shape[1]; j++) {
                line += this.get(i, j) + ', ';
            }
            console.log(line);
        }
        console.log(']');
    }
}
exports.Matrix = Matrix;
//# sourceMappingURL=matrix.js.map