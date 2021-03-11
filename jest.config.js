module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testRegex: [
    './jest_test/booleanCheck.test.ts',
    // './jest_test/common.test.ts'
  ],
  collectCoverage: false
};