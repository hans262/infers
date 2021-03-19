module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testRegex: [
    './test/point.test.ts',
    './test/edge.test.ts',
    './test/common.test.ts',
    './test/matrix.test.ts'
  ],
  collectCoverage: false
}