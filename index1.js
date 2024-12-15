import { readFile } from 'node:fs/promises';
import * as tf from '@tensorflow/tfjs-node';

const NBCDiscrete = {
    alpha: 1,
    unLabels: [], // unique class unLabels (типа + - yess noo...)
    yCounts: {},  // Y count - class counter
    featureCounts: [],
    classFeatureCounts: {}, // X1, X2, X3...
    logPriorProbs: {},
    logConditionalProbs: {},

    setAlpha(value) {
        this.alpha = value;
    },

    //uczenie
    fit(X, y) { // X - observation matrix, y - unLabels
        this.unLabels = [...new Set(y)];
        const totalSamples = y.length;
        this.yCounts = {};
        this.featureCounts = Array(X[0].length).fill(null).map(() => ({}));
        this.classFeatureCounts = {};

        for (const label of this.unLabels) {
            this.yCounts[label] = 0;
            this.classFeatureCounts[label] = Array(X[0].length).fill(null).map(() => ({}));
        }

        // frequency of each class and each feature value
        for (let i = 0; i < X.length; i++) {
            const xi = X[i];
            const yi = y[i];
            this.yCounts[yi]++;
            for (let index = 0; index < xi.length; index++) {
                const value = xi[index];
                this.classFeatureCounts[yi][index][value] = (this.classFeatureCounts[yi][index][value] || 0) + 1;
                this.featureCounts[index][value] = (this.featureCounts[index][value] || 0) + 1;
            }
        }

        this.logPriorProbs = {};
        // P(c)
        for (const label of this.unLabels) {
            this.logPriorProbs[label] = Math.log(this.yCounts[label] / totalSamples);
        }

        this.logConditionalProbs = {};
        // P(x|c)
        for (const label of this.unLabels) {
            this.logConditionalProbs[label] = Array(X[0].length).fill(null).map(() => ({}));
            for (let featureIdx = 0; featureIdx < this.classFeatureCounts[label].length; featureIdx++) {
                const featureCount = this.classFeatureCounts[label][featureIdx];
                const totalLabelCount = this.yCounts[label];
                const uniqueFeatureValues = Object.keys(this.featureCounts[featureIdx]).length;

                for (const [value, count] of Object.entries(featureCount)) {
                     const znamenatel = count + this.alpha;
                    const chislitel = totalLabelCount + uniqueFeatureValues * this.alpha;
                    this.logConditionalProbs[label][featureIdx][value] = Math.log(znamenatel / chislitel);
                }
            }
        }
    },

    getRelativeProbs(X) {
        return X.map(sample => {
            return this.unLabels.map(label => {
                let logProb = this.logPriorProbs[label];
                for (let index = 0; index < sample.length; index++) {
                    const value = sample[index];
                    if (this.logConditionalProbs[label][index][value] !== undefined) {
                        logProb += this.logConditionalProbs[label][index][value];
                    }
                }
                return logProb;
            });
        });
    },

    predict(X) {
        const probs = this.getRelativeProbs(X);
        return probs.map(probArr => {
            const maxIndex = probArr.indexOf(Math.max(...probArr));
            return this.unLabels[maxIndex];
        });
    },

     predictProba(X) {
        const probs = this.getRelativeProbs(X);
        return probs.map(probArr => {
            const maxLogProb = Math.max(...probArr);
            const expProbs = probArr.map(logProb => Math.exp(logProb - maxLogProb));
            const sumExpProbs = expProbs.reduce((sum, val) => sum + val, 0);
            return expProbs.map(expProb => expProb / sumExpProbs);
        });
    }
};


const NBCDiscreteSafe = {
    alpha: 1,
    unLabels: [],
    yCounts: {},
    featureCounts: [],
    classFeatureCounts: {},
    logPriorProbs: {},
    logConditionalProbs: {},

     setAlpha(value) {
        this.alpha = value;
    },

    fit(X, y) {
        this.unLabels = [...new Set(y)];
        const totalSamples = y.length;
        this.yCounts = {};
        this.featureCounts = Array(X[0].length).fill(null).map(() => ({}));
        this.classFeatureCounts = {};

        // P(c)
        for (const label of this.unLabels) {
            this.yCounts[label] = 0;
            this.classFeatureCounts[label] = Array(X[0].length).fill(null).map(() => ({}));
        }

        for (let i = 0; i < X.length; i++) {
            const xi = X[i];
            const yi = y[i];
            this.yCounts[yi]++;
            for (let index = 0; index < xi.length; index++) {
                const value = xi[index];
                this.classFeatureCounts[yi][index][value] = (this.classFeatureCounts[yi][index][value] || 0) + 1;
                this.featureCounts[index][value] = (this.featureCounts[index][value] || 0) + 1;
            }
        }

        this.logPriorProbs = {};
        for (const label of this.unLabels) {
            this.logPriorProbs[label] = Math.log(this.yCounts[label] / totalSamples);
        }

        this.logConditionalProbs = {};
        // P(x|c)
        for (const label of this.unLabels) {
            this.logConditionalProbs[label] = Array(X[0].length).fill(null).map(() => ({}));
            for (let featureIdx = 0; featureIdx < this.classFeatureCounts[label].length; featureIdx++) {
                const featureCount = this.classFeatureCounts[label][featureIdx];
                const totalLabelCount = this.yCounts[label];
                const uniqueFeatureValues = Object.keys(this.featureCounts[featureIdx]).length;

                for (const [value, count] of Object.entries(featureCount)) {
                    const znamenatel = count + this.alpha;
                    const chislitel = totalLabelCount + uniqueFeatureValues * this.alpha;
                    this.logConditionalProbs[label][featureIdx][value] = Math.log(znamenatel / chislitel);

                }
                 //if a value is missing during fit
                for (const value in this.featureCounts[featureIdx]) {
                    const conditionalProbIsCalculated =  this.logConditionalProbs[label][featureIdx][value] === undefined;
                    if (conditionalProbIsCalculated) { // if it is not calc - it meams that we face wit it for the 1st time
                        const znamenatel = this.alpha;
                        const chislitel = totalLabelCount + uniqueFeatureValues * this.alpha;
                        // P(x|c) = (count + alpha) / (totalLabelCount + uniqueFeatureValues * alpha)
                        this.logConditionalProbs[label][featureIdx][value] = Math.log(znamenatel / chislitel);
                    }
                }
             }
         }
    },
    
      getRelativeProbs(X) {
        return X.map(sample => {
            return this.unLabels.map(label => {
                let logProb = this.logPriorProbs[label];
                for (let index = 0; index < sample.length; index++) {
                    const value = sample[index];
                    const isObjectWhereWeStoreLogs = this.logConditionalProbs[label][index][value] !== undefined;
                   if (isObjectWhereWeStoreLogs) {
                       logProb += this.logConditionalProbs[label][index][value];
                   }
                }
                 return logProb;
            });
        });
    },

    predict(X) {
        const probs = this.getRelativeProbs(X);
        return probs.map(probArr => {
            const maxIndex = probArr.indexOf(Math.max(...probArr));
            return this.unLabels[maxIndex];
        });
    },

    predictProba(X) {
       const probs = this.getRelativeProbs(X);
        return probs.map(probArr => {
            const maxLogProb = Math.max(...probArr);
            const expProbs = probArr.map(logProb => Math.exp(logProb - maxLogProb));
            const sumExpProbs = expProbs.reduce((sum, val) => sum + val, 0);
            return expProbs.map(expProb => expProb / sumExpProbs);
        });
    }
};

function get_accuracy(y_expected, y_actual) {
    let hits = 0;
    for (let i = 0; i < y_expected.length; i++) {
        if (y_expected[i] === y_actual[i]) hits += 1;
    }
    return 100.0 * hits / y_actual.length;
}

function discretize_data(data, n_bins = 8) {
    const n_features = data[0].length;
    const discretized_data = data.map(row => row.slice());
    for (let feature_index = 0; feature_index < n_features; feature_index++) {
        const feature_values = data.map(row => row[feature_index]);
        const minVal = Math.min(...feature_values);
        const maxVal = Math.max(...feature_values);
        const binWidth = (maxVal - minVal) / n_bins;
        for (let i = 0; i < discretized_data.length; i++) {
            const value = discretized_data[i][feature_index];
            const binNumber = Math.floor((value - minVal) / binWidth);
            discretized_data[i][feature_index] = Math.min(Math.max(binNumber, 0), n_bins - 1);
        }
    }
    return discretized_data;
}

// Function to split data into train and test sets
function train_test_split(X, y, test_size = 0.3, random_state = 42) {
    const data = X.map((xi, index) => ({ X: xi, y: y[index] }));
    const randomTensor = tf.randomUniform([data.length], 0, 1, 'float32', random_state);
    const randomValues = randomTensor.arraySync();
    const dataWithRandom = data.map((item, index) => ({ ...item, rand: randomValues[index] }));
    dataWithRandom.sort((a, b) => a.rand - b.rand);
    const shuffledData = dataWithRandom.map(({ rand, ...rest }) => rest);
    const testCount = Math.floor(test_size * data.length);
    const testData = shuffledData.slice(0, testCount);
    const trainData = shuffledData.slice(testCount);
    const Xtrain = trainData.map((obj) => obj.X);
    const Ytrain = trainData.map((obj) => obj.y);
    const Xtest = testData.map((obj) => obj.X);
    const Ytest = testData.map((obj) => obj.y);
    return { Xtrain, Xtest, Ytrain, Ytest };
}

function duplicate_features(X, multiplier) {
  const duplicatedX = [];
    for (const row of X) {
        const duplicatedRow = [];
        for (let i = 0; i < multiplier; i++) {
            duplicatedRow.push(...row);
        }
        duplicatedX.push(duplicatedRow);
    }
    return duplicatedX;
}

(async function main() {
    const rawData = await readFile('wine.data', 'utf-8');
    const dataLines = rawData.trim().split('\n');
    const data = dataLines.map(line => line.trim().split(',').map(Number));

    const X = data.map(row => row.slice(1));
    const y = data.map(row => row[0]);

    const binsArray = [3, 5, 7, 10, 20, 100];
    const multipliers = [1, 10, 100];

    for (const multiplier of multipliers) {
        const duplicated_X = duplicate_features(X, multiplier);
        console.log(`Features Multiplier: ${multiplier}`);
        console.log("Bez poprawki Laplace");
    const noLaFlameResults = [];
    for (const bins of binsArray) {
        const discrete_X = discretize_data(duplicated_X, bins);
        const {Xtrain, Xtest, Ytrain, Ytest} = train_test_split(discrete_X, y, 0.3, 736412);

        NBCDiscrete.setAlpha(0);
        NBCDiscrete.fit(Xtrain, Ytrain);
        const Ypred = NBCDiscrete.predict(Xtest);
        const accuracy = get_accuracy(Ytest, Ypred);
        noLaFlameResults.push({
            Bins: bins,
            NBCDiscrete: `${accuracy.toFixed(16)}%`
        });
    }

    console.table(noLaFlameResults);

    console.log("Z poprawką Laplac");
    const laplaceResults = [];
      for (const bins of binsArray) {
        const discrete_X = discretize_data(duplicated_X, bins);
         const { Xtrain, Xtest, Ytrain, Ytest } = train_test_split(discrete_X, y, 0.3, 736412);

        NBCDiscreteSafe.setAlpha(1);
        NBCDiscreteSafe.fit(Xtrain, Ytrain);
        const Ypred = NBCDiscreteSafe.predict(Xtest);
         const accuracy = get_accuracy(Ytest, Ypred);
         laplaceResults.push({
            Bins: bins,
            NBCDiscrete: `${accuracy.toFixed(16)}%`
        });
    }
    console.table(laplaceResults);
    }
})();