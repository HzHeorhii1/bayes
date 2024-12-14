import { readFile } from 'node:fs/promises';

const NBCForNursery = {
    alpha: 1,
    labels: [], // unique labels
    totalCount: 0,
    labelCounts: {}, // Y count - class counter
    featureCounts: [],
    classFeatureCounts: {},
    logConditionalProbs: {},
    logPriorProbs: {},
    logConditionalProbsWithoutLaplace: {},

    setAlpha(value) {
        this.alpha = value;
    },

    //uczenie
    fit(X, y) { // X - observation matrix, y - labels
        this.labels = [...new Set(y)];
        this.totalCount = y.length;
        this.labelCounts = {};
        this.featureCounts = Array(X[0].length).fill(null).map(() => ({}));
        this.classFeatureCounts = {};
        this.logConditionalProbs = {};
        this.logConditionalProbsWithoutLaplace = {};


        for (const label of this.labels) {
            this.labelCounts[label] = 0;
            this.classFeatureCounts[label] = Array(X[0].length).fill(null).map(() => ({}));
        }

        // frequency of each class and each feature value
        for (let i = 0; i < X.length; i++) {
            const xi = X[i];
            const yi = y[i];
            this.labelCounts[yi]++;
            for (let index = 0; index < xi.length; index++) {
                const value = xi[index];
                this.classFeatureCounts[yi][index][value] = (this.classFeatureCounts[yi][index][value] || 0) + 1;
                this.featureCounts[index][value] = (this.featureCounts[index][value] || 0) + 1;
            }
        }

        // calculate log conditional probabilities
        // P(x|c) = (classFeatureCounts(x, c) + alpha) / (classFeatureCounts(c) + alpha * |X|)
        for (const label of this.labels) {
             this.logConditionalProbs[label] = Array(X[0].length).fill(null).map(() => ({}));
             this.logConditionalProbsWithoutLaplace[label] = Array(X[0].length).fill(null).map(() => ({}));
            for (let featureIdx = 0; featureIdx < this.classFeatureCounts[label].length; featureIdx++) {
                const featureCount = this.classFeatureCounts[label][featureIdx];
                const totalLabelCount = this.labelCounts[label];
                const uniqueFeatureValues = Object.keys(this.featureCounts[featureIdx]).length;
                 for (const [value, count] of Object.entries(featureCount)) {
                    let numerator = count + this.alpha;
                    let denominator = totalLabelCount + uniqueFeatureValues * this.alpha;
                    this.logConditionalProbs[label][featureIdx][value] = Math.log(numerator / denominator);
                     numerator = count;
                     denominator = totalLabelCount
                     this.logConditionalProbsWithoutLaplace[label][featureIdx][value] =  denominator > 0 ? Math.log(numerator / denominator): Number.NEGATIVE_INFINITY;
                }
             }
        }

        // calculate log prior probabilities
        // P(c) = count(c) / count(all)
        this.logPriorProbs = {};
        for (const label of this.labels) {
            this.logPriorProbs[label] = Math.log(this.labelCounts[label] / this.totalCount);
        }
    },

    predictProba(X, useLaplace = true) {
       return X.map((xi) => {
            const logProbs = this.labels.map((label) => {
                let logProb = this.logPriorProbs[label];
                for (let index = 0; index < xi.length; index++) {
                    const value = xi[index];
                    if (useLaplace){
                       const isObjectWhereWeStoreLogs = this.logConditionalProbs[label][index][value] !== undefined;
                       logProb += (isObjectWhereWeStoreLogs) ? this.logConditionalProbs[label][index][value] : Math.log(this.alpha / (this.labelCounts[label] + Object.keys(this.featureCounts[index]).length* this.alpha));
                   }
                   else {
                    const isObjectWhereWeStoreLogsWithoutLaplaceNotUndefined = this.logConditionalProbsWithoutLaplace[label][index][value] !== undefined;
                    logProb += (isObjectWhereWeStoreLogsWithoutLaplaceNotUndefined) ? this.logConditionalProbsWithoutLaplace[label][index][value] : Number.NEGATIVE_INFINITY;
                   }
                }
                return logProb;
            });

            const maxLogProb = Math.max(...logProbs);
            const normalizedProbs = logProbs.map(logProb => Math.exp(logProb - maxLogProb));
            const totalProb = normalizedProbs.reduce((sum, p) => sum + p, 0);
            return normalizedProbs.map(p => p / totalProb);
        });
    },

    // P(value | classLabel) = (classFeatureCounts(value) + alpha) / (labelCounts + alpha * |X|)
    calculateConditionalProbability(attributeIndex, value, classLabel, useLaplace = true) {
        const featureCounts = this.classFeatureCounts[classLabel][attributeIndex];
        const labelCount = this.labelCounts[classLabel];
        const uniqueFeatureValues = Object.keys(this.featureCounts[attributeIndex]).length;

        const numerator = (featureCounts[value] || 0) + (useLaplace ? this.alpha : 0);
        const denominator = labelCount + (useLaplace ? uniqueFeatureValues * this.alpha : 0);

        return denominator > 0 ? numerator / denominator : 0;
    },

    //klasyfikowanie(it searches a class with the highest probability)
    predict(X, useLaplace = true) {
      const probas = this.predictProba(X, useLaplace);
        return probas.map((probs) => {
            const maxProbIndex = probs.indexOf(Math.max(...probs));
            return this.labels[maxProbIndex];
        });
    }
};

const rawData = await readFile('nursery.data', 'utf-8');
const attributes = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class'];
const rows = rawData.trim().split('\n').map(line => line.split(','));

// non-numeric encoding to numeric
const encoders = {};
for (const attr of attributes) {
    encoders[attr] = {};
}
for (const row of rows) {
    for (let colIdx = 0; colIdx < row.length; colIdx++) {
        const value = row[colIdx];
        const attr = attributes[colIdx];
        const isValueEncoded = value in encoders[attr];
        if (!isValueEncoded) encoders[attr][value] = Object.keys(encoders[attr]).length;
    }
}

const encodedData = rows.map(row => row.map((value, colIdx) => encoders[attributes[colIdx]][value]));

// separationo of X and y
const X = encodedData.map(row => row.slice(0, -1));
const y = encodedData.map(row => row[row.length - 1]);


// Experiment with different number of bins for each feature (original + duplicated) 
const numBinsOptions = [3, 5, 7];
const originalFeatureCount = X[0].length;

// Function to duplicate features by multiplying them by 2, 3, 4, etc.
function duplicateFeatures(X, numDuplicates) {
    const duplicatedX = X.map(row => {
        let newRow = [...row];
        for (let i = 0; i < numDuplicates; i++) {
            newRow = newRow.concat(row.map(x => x * (i + 2)));
        }
        return newRow
    });
    return duplicatedX;
}

const testX = X;

const { MultinomialNB } = await import('ml-naivebayes');
function calculateAccuracy(y_true, y_pred) {
    let correct = 0;
    for (let i = 0; i < y_true.length; i++) {
        const isPredictedTrue = y_true[i] === y_pred[i];
        if (isPredictedTrue) correct++;
    }
    return (correct / y_true.length) * 100;
}


console.log("With laplace:")
const laplaceResults = [];

for (const numBins of numBinsOptions) {
    const duplicatedX = duplicateFeatures(testX, numBins);
    const nbc = Object.create(NBCForNursery);
    nbc.setAlpha(1);
    nbc.fit(duplicatedX, y);
    const predicted_y = nbc.predict(duplicatedX, true);
    const accuracyNBC = calculateAccuracy(y, predicted_y);

    const modelMNB = new MultinomialNB();
    modelMNB.train(duplicatedX, y);
    const predicted_yMNB = modelMNB.predict(duplicatedX);
    const accuracyMNB = calculateAccuracy(y, predicted_yMNB)

    laplaceResults.push({
        Bins: numBins,
        naїveBayes: `${accuracyNBC}%`,
        MultinomialNB: `${accuracyMNB}%`
    });
}

console.table(laplaceResults);


console.log('---------------------------------');
console.log("Without laplace:")
const noLaFlameResults = [];

for (const numBins of numBinsOptions) {
    const duplicatedX = duplicateFeatures(X, numBins);
    const nbc = Object.create(NBCForNursery);
    nbc.setAlpha(1);
    nbc.fit(duplicatedX, y);
    const predicted_y = nbc.predict(duplicatedX, false);
    const accuracyNBC = calculateAccuracy(y, predicted_y);

    const modelMNB = new MultinomialNB();
    modelMNB.train(duplicatedX, y);
    const predicted_yMNB = modelMNB.predict(duplicatedX);
    const accuracyMNB = calculateAccuracy(y, predicted_yMNB)

    noLaFlameResults.push({
        Bins: numBins,
        naїveBayes: `${accuracyNBC}%`,
        MultinomialNB: `${accuracyMNB}%`
    });
}

console.table(noLaFlameResults);

// decoding classes to original values
const decisionClasses = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"];
const encodedClasses = {};
for (const cls of decisionClasses) {
    encodedClasses[cls] = encoders.class[cls];
}

const attributeValues = {
    'parents': 'usual',
    'has_nurs': 'less_proper',
    'form': 'incomplete',
    'children': '2',
    'housing': 'less_conv',
    'finance': 'convenient',
    'social': 'problematic',
    'health': 'recommended'
};

// calculate conditional probabilities for each class 
const conditionalProbabilitiesWith = {};
const conditionalProbabilitiesWithout = {};

const nbcTest = Object.create(NBCForNursery);
nbcTest.setAlpha(1);
nbcTest.fit(X, y);
for (const cls of decisionClasses) {
    conditionalProbabilitiesWith[cls] = {};
    conditionalProbabilitiesWithout[cls] = {};
    for (const attr of Object.keys(attributeValues)) {
        const attrIndex = attributes.indexOf(attr);
        const valueEncoded = encoders[attr][attributeValues[attr]];
        conditionalProbabilitiesWith[cls][attr] = nbcTest.calculateConditionalProbability(attrIndex, valueEncoded, encodedClasses[cls], true);
        conditionalProbabilitiesWithout[cls][attr] = nbcTest.calculateConditionalProbability(attrIndex, valueEncoded, encodedClasses[cls], false);
    }
}

const priorProbabilities = {};
for (const cls of decisionClasses) {
    priorProbabilities[cls] = nbcTest.labelCounts[encodedClasses[cls]] / nbcTest.totalCount;
}

const probWithoutLaplace = {};
const probWithLaplace = {};
for (const cls of decisionClasses) {
    probWithoutLaplace[cls] = priorProbabilities[cls];
    probWithLaplace[cls] = priorProbabilities[cls];
    for (const attr of Object.keys(attributeValues)) {
        probWithoutLaplace[cls] *= conditionalProbabilitiesWithout[cls][attr];
        probWithLaplace[cls] *= conditionalProbabilitiesWith[cls][attr];
    }
}

// normalization
const resultsWith = Object.values(probWithLaplace);
const normalizedWith = resultsWith.map(p => p / resultsWith.reduce((sum, x) => sum + x, 0));

console.log("\nWith Laplace:");
console.log(resultsWith);
console.log("W/o Laplace:");
console.log(normalizedWith);