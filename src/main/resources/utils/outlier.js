const fs = require('fs');
const path = require('path');

const pairsWithFeatures = fs.readFileSync('../pairs_with_features.csv', 'utf8').split('\n').slice(1);
const pairs_outliers = pairsWithFeatures.map(pair => {
  pair = pair.split(',')
  const res = []
  for (const i of [2, 3, 4, 5, 6]) res.push(parseFloat(pair[i]));
  return res.join(',');
});
fs.writeFileSync(path.join(__dirname, '../pairs_outliers.txt'), pairs_outliers.join('\n'));

const pairs_cross_validation = pairsWithFeatures.map(pair => {
  pair = pair.split(',')
  const res = [parseInt(pair[7])];
  for (const i of [2, 3, 4, 5, 6]) res.push(parseFloat(pair[i]));
  return res.join(',');
});
fs.writeFileSync(path.join(__dirname, '../pairs_cross_validation.txt'), pairs_cross_validation.join('\n'));
