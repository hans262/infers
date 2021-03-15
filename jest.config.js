module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testRegex: [
    './jest_test/point.test.ts',
    './jest_test/edge.test.ts',
    './jest_test/common.test.ts',
    './jest_test/matrix.test.ts'
  ],
  collectCoverage: false
}