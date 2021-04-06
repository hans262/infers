
  // src/matrix.ts
  var Matrix = class {
    constructor(data) {
      let t = data.find((d, i) => data[i - 1] && d.length !== data[i - 1].length);
      if (t)
        throw new Error("\u77E9\u9635\u5217\u4E0D\u6B63\u786E");
      this.shape = [data.length, data[0].length];
      this.self = data;
    }
    slice(start, end) {
      return new Matrix(this.self.slice(start, end));
    }
    connect(b) {
      if (this.shape[1] !== b.shape[1]) {
        throw new Error("\u5217\u6570\u4E0D\u7EDF\u4E00");
      }
      let tmp = this.dataSync().concat(b.dataSync());
      return new Matrix(tmp);
    }
    zeroed() {
      return this.atomicOperation((_) => 0);
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
        case "+=":
          this.self[row][col] += val;
          break;
        case "-=":
          this.self[row][col] -= val;
          break;
        case "*=":
          this.self[row][col] *= val;
          break;
        case "/=":
          this.self[row][col] /= val;
          break;
        default:
          this.self[row][col] = val;
      }
    }
    expand(n, position) {
      let m = [];
      for (let i = 0; i < this.shape[0]; i++) {
        if (position === "L") {
          m.push([n, ...this.getRow(i)]);
        } else {
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
        throw new Error("\u53EA\u6709\u65B9\u9635\u624D\u80FD\u8BA1\u7B97\u884C\u5217\u5F0F");
      }
      if (this.shape[0] === 2 && this.shape[1] === 2) {
        return this.get(0, 0) * this.get(1, 1) - this.get(0, 1) * this.get(1, 0);
      } else {
        let m = 0;
        for (let i = 0; i < this.shape[1]; i++) {
          if (this.get(0, i) !== 0) {
            m += this.get(0, i) * (-1) ** (i + 2) * this.cominor(0, i).det();
          }
        }
        return m;
      }
    }
    cominor(rowi, coli) {
      if (this.shape[0] < 2 || this.shape[1] < 2) {
        throw new Error("\u6C42\u4F59\u5B50\u5F0F\u884C\u548C\u5217\u5FC5\u987B\u5927\u4E8E2\u624D\u6709\u610F\u4E49");
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
        throw new Error("\u5FC5\u987B\u6EE1\u8DB3\u4E24\u4E2A\u77E9\u9635\u662F\u540C\u5F62\u77E9\u9635");
      }
      let n = [];
      for (let i = 0; i < this.shape[0]; i++) {
        let m = [];
        for (let j = 0; j < this.shape[1]; j++) {
          let c = oper === "add" ? this.get(i, j) + b.get(i, j) : this.get(i, j) - b.get(i, j);
          m.push(c);
        }
        n.push(m);
      }
      return new Matrix(n);
    }
    subtraction(b) {
      return this.coLocationOperation(b, "sub");
    }
    addition(b) {
      return this.coLocationOperation(b, "add");
    }
    numberMultiply(b) {
      return this.atomicOperation((item) => item * b);
    }
    multiply(b) {
      if (this.shape[1] !== b.shape[0]) {
        throw new Error("\u5F53\u77E9\u9635A\u7684\u5217\u6570\u7B49\u4E8E\u77E9\u9635B\u7684\u884C\u6570\uFF0CA\u4E0EB\u624D\u53EF\u4EE5\u76F8\u4E58");
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
        const average = min + range / 2;
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
        let line = " ";
        for (let j = 0; j < this.shape[1]; j++) {
          line += this.get(i, j) + ", ";
        }
        console.log(line);
      }
      console.log("]");
    }
  };

  // src/graphical.ts
  var Point = class {
    constructor(X, Y) {
      this.X = X;
      this.Y = Y;
    }
    contrast(pt) {
      return this.X === pt.X && this.Y === pt.Y;
    }
  };
  var Edge = class {
    constructor(cond1, cond2) {
      this.start = new Point(cond1[0], cond1[1]);
      this.end = new Point(cond2[0], cond2[1]);
      if (this.start.contrast(this.end)) {
        throw new Error("\u4E24\u4E2A\u70B9\u4E0D\u80FD\u76F8\u540C");
      }
    }
    minXY() {
      let X = Math.min(this.start.X, this.end.X);
      let Y = Math.min(this.start.Y, this.end.Y);
      return new Point(X, Y);
    }
    maxXY() {
      let X = Math.max(this.start.X, this.end.X);
      let Y = Math.max(this.start.Y, this.end.Y);
      return new Point(X, Y);
    }
    testPointIn(pt) {
      if (pt.contrast(this.start) || pt.contrast(this.end)) {
        return true;
      }
      let slope1 = pt.X - this.start.X === 0 ? Infinity : (pt.Y - this.start.Y) / (pt.X - this.start.X);
      let slope2 = pt.X - this.end.X === 0 ? Infinity : (pt.Y - this.end.Y) / (pt.X - this.end.X);
      return slope1 === slope2;
    }
    testPointInside(pt) {
      if (this.testPointIn(pt)) {
        let min = this.minXY();
        let max = this.maxXY();
        return pt.X >= min.X && pt.X <= max.X && (pt.Y >= min.Y && pt.Y <= max.Y);
      }
      return false;
    }
  };
  var Path = class {
    constructor(pts) {
      this.pts = pts;
      if (pts.length < 2) {
        throw new Error("\u81F3\u5C11\u4E24\u4E2A\u70B9");
      }
      const n = pts.find((pt, i) => {
        const next = pts[i + 1];
        if (!next)
          return false;
        return pt.contrast(next);
      });
      if (n) {
        throw new Error("\u4E0D\u80FD\u6709\u8FDE\u7EED\u91CD\u5408\u7684\u70B9");
      }
    }
  };
  var Polygon = class {
    constructor(pts) {
      this.points = [];
      for (let i = 0; i < pts.length; i++) {
        this.points.push(new Point(pts[i][0], pts[i][1]));
      }
      if (this.points.length < 3) {
        throw new Error("\u81F3\u5C11\u4E09\u4E2A\u70B9");
      }
      const r0 = this.points.map((p) => p.X.toString() + p.Y.toString()).sort();
      const r1 = r0.find((x, i) => x === r0[i + 1]);
      if (r1) {
        throw new Error("\u4E0D\u80FD\u6709\u76F8\u540C\u7684\u70B9");
      }
      const first = this.points[0];
      const r3 = this.points.slice(1);
      let m = r3.map((p) => p.X === first.X ? Infinity : (p.Y - first.Y) / (p.X - first.X));
      if (new Set(m).size === 1) {
        throw new Error("\u6240\u6709\u70B9\u4E0D\u80FD\u5728\u4E00\u6761\u7EBF\u4E0A");
      }
    }
    testPointInsidePolygon(pt) {
      let polygon = this.points;
      var result = 0, cnt = polygon.length;
      if (cnt < 3)
        return 0;
      var ip = polygon[0];
      for (var i = 1; i <= cnt; ++i) {
        var ipNext = i === cnt ? polygon[0] : polygon[i];
        if (ipNext.Y === pt.Y) {
          if (ipNext.X === pt.X || ip.Y === pt.Y && ipNext.X > pt.X === ip.X < pt.X)
            return -1;
        }
        if (ip.Y < pt.Y !== ipNext.Y < pt.Y) {
          if (ip.X >= pt.X) {
            if (ipNext.X > pt.X)
              result = 1 - result;
            else {
              var d = (ip.X - pt.X) * (ipNext.Y - pt.Y) - (ipNext.X - pt.X) * (ip.Y - pt.Y);
              if (d === 0)
                return -1;
              else if (d > 0 === ipNext.Y > ip.Y)
                result = 1 - result;
            }
          } else {
            if (ipNext.X > pt.X) {
              var d = (ip.X - pt.X) * (ipNext.Y - pt.Y) - (ipNext.X - pt.X) * (ip.Y - pt.Y);
              if (d === 0)
                return -1;
              else if (d > 0 === ipNext.Y > ip.Y)
                result = 1 - result;
            }
          }
        }
        ip = ipNext;
      }
      return result;
    }
  };

  // src/common.ts
  function toFixed(num, fix) {
    const amount = 10 ** fix;
    return ~~(num * amount) / amount;
  }

  // src/BPNet.ts
  var BPNet = class {
    constructor(shape, conf) {
      this.shape = shape;
      this.mode = "sgd";
      this.rate = 0.01;
      if (shape.length < 2) {
        throw new Error("The network has at least two layers");
      }
      this.nlayer = shape.length;
      const [w, b] = this.initwb();
      this.w = w;
      this.b = b;
      if (conf) {
        if (conf.mode)
          this.mode = conf.mode;
        if (conf.rate)
          this.rate = conf.rate;
        if (conf.w)
          this.w = conf.w;
        if (conf.b)
          this.b = conf.b;
        if (conf.scalem)
          this.scalem = conf.scalem;
      }
    }
    nOfLayer(l) {
      let n = this.shape[l];
      return Array.isArray(n) ? n[0] : n;
    }
    afOfLayer(l) {
      let n = this.shape[l];
      return Array.isArray(n) ? n[1] : void 0;
    }
    initwb(v) {
      let w = [];
      let b = [];
      for (let l = 1; l < this.shape.length; l++) {
        w[l] = Matrix.generate(this.nOfLayer(l), this.nOfLayer(l - 1), v);
        b[l] = Matrix.generate(1, this.nOfLayer(l), v);
      }
      return [w, b];
    }
    afn(x, l, rows) {
      let af = this.afOfLayer(l);
      switch (af) {
        case "Sigmoid":
          return 1 / (1 + Math.exp(-x));
        case "Relu":
          return x >= 0 ? x : 0;
        case "Tanh":
          return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
        case "Softmax":
          let d = Math.max(...rows);
          return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0);
        default:
          return x;
      }
    }
    afd(x, l) {
      let af = this.afOfLayer(l);
      switch (af) {
        case "Sigmoid":
          return x * (1 - x);
        case "Relu":
          return x >= 0 ? 1 : 0;
        case "Tanh":
          return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2;
        case "Softmax":
        default:
          return 1;
      }
    }
    calcnet(xs) {
      let hy = [];
      for (let l = 0; l < this.nlayer; l++) {
        if (l === 0) {
          hy[l] = xs;
          continue;
        }
        let a = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j));
        hy[l] = a.atomicOperation((item, i) => this.afn(item, l, a.getRow(i)));
      }
      return hy;
    }
    zoomScalem(xs) {
      return xs.atomicOperation((item, _, j) => {
        if (!this.scalem)
          return item;
        return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j);
      });
    }
    predict(xs) {
      if (xs.shape[1] !== this.nOfLayer(0)) {
        throw new Error(`\u7279\u5F81\u4E0E\u7F51\u7EDC\u8F93\u5165\u4E0D\u7B26\u5408\uFF0Cinput num -> ${this.nOfLayer(0)}`);
      }
      return this.calcnet(this.zoomScalem(xs))[this.nlayer - 1];
    }
    calcDerivativeMul(hy, ys) {
      let m = ys.shape[0];
      let dws = null;
      let dys = null;
      for (let n = 0; n < m; n++) {
        let nhy = hy.map((item) => new Matrix([item.getRow(n)]));
        let nys = new Matrix([ys.getRow(n)]);
        let {dw, dy} = this.calcDerivative(nhy, nys);
        dws = dws ? dws.map((d, l) => d.addition(dw[l])) : dw;
        dys = dys ? dys.map((d, l) => d.addition(dy[l])) : dy;
      }
      dws = dws.map((d) => d.atomicOperation((item) => item / m));
      dys = dys.map((d) => d.atomicOperation((item) => item / m));
      return {dy: dys, dw: dws};
    }
    calcDerivative(hy, ys) {
      let dw = this.w.map((w) => w.zeroed());
      let dy = this.b.map((b) => b.zeroed());
      for (let l = this.nlayer - 1; l > 0; l--) {
        if (l === this.nlayer - 1) {
          for (let j = 0; j < this.nOfLayer(l); j++) {
            dy[l].update(0, j, (hy[l].get(0, j) - ys.get(0, j)) * this.afd(hy[l].get(0, j), l));
            for (let k = 0; k < this.nOfLayer(l - 1); k++) {
              dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j));
            }
          }
          continue;
        }
        for (let j = 0; j < this.nOfLayer(l); j++) {
          for (let i = 0; i < this.nOfLayer(l + 1); i++) {
            dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), "+=");
          }
          dy[l].update(0, j, this.afd(hy[l].get(0, j), l), "*=");
          for (let k = 0; k < this.nOfLayer(l - 1); k++) {
            dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j));
          }
        }
      }
      return {dy, dw};
    }
    update(dy, dw) {
      for (let l = 1; l < this.nlayer; l++) {
        this.w[l] = this.w[l].subtraction(dw[l].numberMultiply(this.rate));
        this.b[l] = this.b[l].subtraction(dy[l].numberMultiply(this.rate));
      }
    }
    cost(hy, ys) {
      let m = ys.shape[0];
      let sub = hy.subtraction(ys).atomicOperation((item) => item ** 2).columnSum();
      let tmp = sub.getRow(0).map((v) => v / (2 * m));
      return tmp.reduce((p, c) => p + c) / tmp.length;
    }
    bgd(xs, ys, conf) {
      for (let ep = 0; ep < conf.epochs; ep++) {
        let hy = this.calcnet(xs);
        let {dy, dw} = this.calcDerivativeMul(hy, ys);
        this.update(dy, dw);
        if (conf.onEpoch)
          conf.onEpoch(ep, this.cost(hy[this.nlayer - 1], ys));
      }
    }
    sgd(xs, ys, conf) {
      let m = ys.shape[0];
      for (let ep = 0; ep < conf.epochs; ep++) {
        let hys = null;
        for (let n = 0; n < m; n++) {
          let xss = new Matrix([xs.getRow(n)]);
          let yss = new Matrix([ys.getRow(n)]);
          let hy = this.calcnet(xss);
          const {dy, dw} = this.calcDerivative(hy, yss);
          this.update(dy, dw);
          hys = hys ? hys.connect(hy[this.nlayer - 1]) : hy[this.nlayer - 1];
        }
        if (conf.onEpoch)
          conf.onEpoch(ep, this.cost(hys, ys));
      }
    }
    mbgd(xs, ys, conf) {
      let m = ys.shape[0];
      let batchSize = conf.batchSize ? conf.batchSize : 10;
      let batch = Math.ceil(m / batchSize);
      for (let ep = 0; ep < conf.epochs; ep++) {
        let {xs: xst, ys: yst} = this.upset(xs, ys);
        let eploss = 0;
        for (let b = 0; b < batch; b++) {
          let start = b * batchSize;
          let end = start + batchSize;
          end = end > m ? m : end;
          let size = end - start;
          let xss = xst.slice(start, end);
          let yss = yst.slice(start, end);
          let hy = this.calcnet(xss);
          const {dy, dw} = this.calcDerivative(hy, yss);
          this.update(dy, dw);
          let bloss = this.cost(hy[this.nlayer - 1], yss);
          eploss += bloss;
          if (conf.onBatch)
            conf.onBatch(b, size, bloss);
        }
        if (conf.onEpoch)
          conf.onEpoch(ep, eploss / batch);
      }
    }
    upset(xs, ys) {
      let xss = xs.dataSync();
      let yss = ys.dataSync();
      for (let i = 1; i < ys.shape[0]; i++) {
        let random = Math.floor(Math.random() * (i + 1));
        [xss[i], xss[random]] = [xss[random], xss[i]];
        [yss[i], yss[random]] = [yss[random], yss[i]];
      }
      return {xs: new Matrix(xss), ys: new Matrix(yss)};
    }
    fit(xs, ys, conf) {
      if (xs.shape[0] !== ys.shape[0]) {
        throw new Error("\u8F93\u5165\u8F93\u51FA\u77E9\u9635\u884C\u6570\u4E0D\u7EDF\u4E00");
      }
      if (xs.shape[1] !== this.nOfLayer(0)) {
        throw new Error(`\u7279\u5F81\u4E0E\u7F51\u7EDC\u8F93\u5165\u4E0D\u7B26\u5408\uFF0Cinput num -> ${this.nOfLayer(0)}`);
      }
      if (ys.shape[1] !== this.nOfLayer(this.nlayer - 1)) {
        throw new Error(`\u6807\u7B7E\u4E0E\u7F51\u7EDC\u8F93\u51FA\u4E0D\u7B26\u5408\uFF0Coutput num -> ${this.nOfLayer(this.nlayer - 1)}`);
      }
      if (conf.batchSize && conf.batchSize > ys.shape[0]) {
        throw new Error(`\u6279\u6B21\u5927\u5C0F\u4E0D\u80FD\u5927\u4E8E\u6837\u672C\u6570`);
      }
      const [nxs, scalem] = xs.normalization();
      this.scalem = scalem;
      xs = nxs;
      switch (this.mode) {
        case "bgd":
          return this.bgd(xs, ys, conf);
        case "mbgd":
          return this.mbgd(xs, ys, conf);
        case "sgd":
        default:
          return this.sgd(xs, ys, conf);
      }
    }
  };

