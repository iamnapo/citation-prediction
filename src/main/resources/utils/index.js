const fs = require('fs');
const path = require('path');

function getRandom(arr, n) {
  var result = new Array(n),
      len = arr.length,
      taken = new Array(len);
  if (n > len)
      throw new RangeError("getRandom: more elements taken than available");
  while (n--) {
      var x = Math.floor(Math.random() * len);
      result[n] = arr[x in taken ? taken[x] : x];
      taken[x] = --len in taken ? taken[len] : len;
  }
  return result;
}

const allEdges = fs.readFileSync('../Cit-HepTh.txt', 'utf8').split('\n').slice(1);
let nodes = [];
allEdges.forEach(el => nodes = nodes.concat(el.split(' ')));
nodes = [...new Set(nodes)];
fs.writeFileSync(path.join(__dirname, '../nodesForPart2.txt'), getRandom(nodes, 500).join('\n'));
