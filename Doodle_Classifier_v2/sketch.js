// Image Classifier using 1000 train, cat, and rainbow doodles from Google's Quick Draw! database
// https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
// Uses p5.js and Tensorflow.js libraries
// Inspied by the doodle classifier made by Daniel Shiffman:
// https://codingtrain.github.io/Toy-Neural-Network-JS/examples/doodle_classification/

const CAT = 0;
const COOKIE = 1;
const TRAIN = 2;
const FLOWER = 3;

let catsData;
let trainsData;
let cookiesData;
let flowersData;

let cats = {};
let trains = {};
let cookies = {};
let flowers = {};

let trainingNetwork = true;

function preload() {
  catsData = loadBytes('data/cats1000.bin');
  trainsData = loadBytes('data/trains1000.bin');
  cookiesData = loadBytes('data/cookies1000.bin');
  flowersData = loadBytes('data/flowers1000.bin');
}

function setup() {
  // Preparing the data
  prepareData(cats, catsData, CAT);
  prepareData(cookies, cookiesData, COOKIE);
  prepareData(trains, trainsData, TRAIN);
  prepareData(flowers, flowersData, FLOWER);

  // Organizing the training data
  let training = [];
  training = training.concat(cats.training);
  training = training.concat(cookies.training);
  training = training.concat(trains.training);
  training = training.concat(flowers.training);
  shuffle(training, true);

  // Organizing the testing data
  let testing = [];
  testing = testing.concat(cats.testing);
  testing = testing.concat(cookies.testing);
  testing = testing.concat(trains.testing);
  testing = testing.concat(flowers.testing);

  // Creating an array of known outputs that matches the shuffled testing data
  let goals = [];
  for (let i = 0; i < training.length; i++) {
    let ans = training[i].label;
    if (ans == 0) {
      res = [1, 0, 0, 0]; // cat
    } else if (ans == 1) {
      res = [0, 1, 0, 0]; // cookie
    } else if (ans == 2) {
      res = [0, 0, 1, 0]; //train
    } else if (ans == 3) {
      res = [0, 0, 0, 1]; // flower
    }
    goals.push(res);
  }

  // Creating the Neural Network
  const network = tf.sequential();

  const hidden1 = tf.layers.dense({
    units: 500,
    inputShape: [784],
    activation: 'sigmoid',
    useBias: true
  });
  network.add(hidden1);


  const output = tf.layers.dense({
    units: 4,
    activation: 'softmax',
    useBias: true
  });

  network.add(output);

  learning_rate = 0.05;
  let opt = tf.train.sgd(learning_rate);

  network.compile({
    optimizer: opt,
    loss: 'meanSquaredError'
  });

  let xs = tf.tensor(training);
  let ys = tf.tensor(goals);

  let answers = [];
  for (let i = 0; i < testing.length; i++) {
    let ans = testing[i].label;
    if (ans == 0) {
      res = [1, 0, 0, 0]; // cat
    } else if (ans == 1) {
      res = [0, 1, 0, 0]; // rainbow
    } else if (ans == 2) {
      res = [0, 0, 1, 0]; //train
    } else if (ans == 3) {
      res = [0, 0, 0, 1];
    }
    answers.push(res);
  }

  let trialxs = [];
  for (let i = 0; i < testing.length; i++) {
    let new_i = tf.tensor2d([testing[i]], [1, 784]);
    trialxs.push(new_i);
  }

  // Training the network and testing
  async function train() {
    for (let i = 0; i < 20; i++) {
      const result = await network.fit(xs, ys, {
        epochs: 1,
        shuffle: true
      });
      console.log(Math.round(Math.pow(10,8) * result.history.loss[0]) / Math.pow(10,8));
    }
  }

  train().then(() => {
    trainingNetwork = false;
    console.log("training done!")
  }).then(() => {
    testMethod();
  });

  const index = Math.round(random(trialxs.length));

  function testMethod() {
    removeElements();
    sample = createButton("Use a sample doodle");
    original = createButton("Make your own doodle");
    sample.mouseReleased(() => test())
    original.mouseReleased(() => myDoodle())
  }

  function myDoodle() {
    loop();
    removeElements();
    createP('Draw your own cat, train, cookie, or flower and see if the neural network knows what it is!');
    createCanvas(280, 280);
    background(255);
    finished = createButton("My doodle is complete!");
    finished.mouseReleased(() => testOriginal())
  }

  function testOriginal() {
    noLoop();
    removeElements();
    let inputs = [];
    let img = get();
    img.resize(28, 28);
    img.loadPixels();
    for (let i = 0; i < 784; i++) {
      let bright = 255 - img.pixels[i * 4];
      inputs.push(bright);
    }
    let newt = tf.tensor2d([inputs], [1, 784]);
    network.predict(newt).print();
    let res1 = network.predict(newt).arraySync();
    let newans = tf.argMax(res1, 1).arraySync();
    let guess;
    let percent;
    let r = res1[0];
    if (newans[0] == 0) {
      guess = 'cat';
      percent = Math.round(100 * (r[0]));
    } else if (newans[0] == 1) {
      guess = 'cookie';
      percent = Math.round(100 * (r[1]));
    } else if (newans[0] == 2) {
      guess = 'train';
      percent = Math.round(100 * (r[2]));
    } else if (newans[0] == 3) {
      guess = 'flower';
      percent = Math.round(100 * (r[3]));
    }
    refresh = createButton('refresh');
    refresh.mouseReleased(() => myDoodle())
    //createP('I am ' + percent + '% confident that your doodle was a ' + guess);
    createP(`I think that your doodle was a ${guess}`);
    createP('Was I correct?');
    yes = createButton("Yes");
    no = createButton("No");
    yes.mouseReleased(() => createP('Yay!'))
    no.mouseReleased(() => createP('Oh no! refresh to try again.'))
  }

  function test() {
    removeElements();
    createCanvas(28, 28); //Display image being tested
    background(0);
    loadPixels();
    scale(4.0);
    for (let i = 0; i < 4 * 784; i++) {
      let col = color(testing[index][i], testing[index][i], testing[index][i], 255);
      let g = 4 * i;
      pixels[g] = red(col);
      pixels[g + 1] = green(col);
      pixels[g + 2] = blue(col);
      pixels[g + 3] = alpha(255);
      updatePixels();
    }
    testing = createButton("test");
    testing.mouseReleased(() => testit())
  }

  function testit() {
    network.predict(trialxs[index]).print();
    console.log(answers[index]);
    removeElements();
    let result1 = network.predict(trialxs[index]).arraySync();
    let id;
    let perc;
    let r = result1[0];
    let itsans = tf.argMax(result1, 1).arraySync();
    if (itsans[0] == 0) {
      id = 'cat';
      perc = Math.round(100 * (r[0]));
    } else if (itsans[0] == 1) {
      id = 'cookie';
      perc = Math.round(100 * (r[1]));
    } else if (itsans[0] == 2) {
      id = 'train';
      perc = Math.round(100 * (r[2]));
    } else if (itsans[0] == 3) {
      id = 'flower';
      perc = Math.round(100 * (r[3]));
    }

    createP(`I am ${perc}% confident that this is a ${id}.`);
    let correct;
    let wrong = false;
    if (itsans[0] == 0 && answers[index][0] == 1) {
      correct = 'correct!';
    } else if (itsans[0] == 1 && answers[index][1] == 1) {
      correct = 'correct!';
    } else if (itsans[0] == 2 && answers[index][2] == 1) {
      correct = 'correct!';
    } else if (itsans[0] == 3 && answers[index][3] == 1) {
      correct = 'correct!';
    } else {
      correct = 'incorrect!';
      wrong = true;
    }

    createP(`I was ${correct}`);

    let realans;
    if (answers[index][0] == 1) {
      realans = 'cat';
    } else if (answers[index][1] == 1) {
      realans = 'cookie';
    } else if (answers[index][2] == 1) {
      realans = 'train';
    } else if (answers[index][3] == 1) {
      realans = 'flower';
    }

    if (wrong) {
      createP(`The real answer was ${realans}.`);
    }
  }

  if (trainingNetwork) {
    createElement('marquee','Neural network is being trained');
  }
}

function draw() {
  strokeWeight(18);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}
