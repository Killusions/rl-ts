import { Engine, Render, World, Bodies, Body, Runner, Events } from 'matter-js';
import { Chart } from 'chart.js/auto'; 

// Initialize Chart
const ctx = document.getElementById('myChart') as HTMLCanvasElement;
const myChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [] as string[], // Explicit type
    datasets: [{
      label: 'Best Score',
      data: [] as number[], // Explicit type
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    },
    {
      label: 'Worst Score',
      data: [] as number[], // Explicit type
      fill: false,
      borderColor: 'rgb(227, 25, 25)',
      tension: 0.1
    }]
  }
});

const ictx = document.getElementById('iterationsChart') as HTMLCanvasElement;
const iterationsChart = new Chart(ictx, {
  type: 'line',
  data: {
    labels: [] as string[], // Explicit type
    datasets: [{
      label: 'Episodes',
      data: [] as number[], // Explicit type
      fill: false,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1
    },
    {
      label: 'Learning rate (Mutation Alpha) (1 / 1000)',
      data: [] as number[], // Explicit type
      fill: false,
      borderColor: 'rgb(227, 25, 25)',
      tension: 0.1
    }]
  }
});

// RL Constants
const ALPHA = 0.1;
const GAMMA = 0.9;
const MAX_EPISODES = 2000;
const EPSILON = 0.2;
const INITIAL_MUTATION_ALPHA = 0.3; // initial mutation rate
const MUTATION_DECAY = 0.99; // decay factor for mutation rate
const RELATIVE_PERFORMANCE_AVERAGE_DURATION = 5; // Adjust based on your specific use case
const RELATIVE_PERFORMANCE_THRESHOLD = 0.9; // Adjust based on your specific use case
let episodeCount = 0;
let iterationsCount = 0;
let bestScore = -Infinity;
let worstScore = Infinity;
const ballStartX = 100;
const ballStartY = 200;
const ballRadius = 20;
const timeScale = 5;

let Q: Map<string, Map<string, number>> = new Map();

// Environment
const engine = Engine.create({
  gravity: {
    x: 0,
    y: 0
  },
  timing: {
    timeScale
  } as any
});
const render = Render.create({
  element: document.getElementById('matterContainer')!,
  engine: engine
});

// Define IDs for different object types
const BALL_CATEGORY = 0x0001;
const WALL_CATEGORY = 0x0002;
const OBSTACLE_CATEGORY = 0x0004;
const TARGET_CATEGORY = 0x0008;

// Create bounding box
const wallOptions = { isStatic: true, collisionFilter: { category: WALL_CATEGORY, mask: BALL_CATEGORY } };
// Define the wall thickness (e.g., 20 units)
const wallThickness = 200;

// Create the walls with extended boundaries
const wallTop = Bodies.rectangle(400, -wallThickness / 2, 800, wallThickness, wallOptions);
const wallBottom = Bodies.rectangle(400, 600 + wallThickness / 2, 800, wallThickness, wallOptions);
const wallLeft = Bodies.rectangle(-wallThickness / 2, 300, wallThickness, 600, wallOptions);
const wallRight = Bodies.rectangle(800 + wallThickness / 2, 300, wallThickness, 600, wallOptions);

// Add the walls to the world
World.add(engine.world, [wallTop, wallBottom, wallLeft, wallRight]);

// Create multiple balls
const numBalls = 10;
const balls: any[] = [];

// For Balls
for (let i = 0; i < numBalls; i++) {
  const ball = Bodies.circle(ballStartX, ballStartY, ballRadius, {
    collisionFilter: {
      category: BALL_CATEGORY,
      mask: WALL_CATEGORY | OBSTACLE_CATEGORY | TARGET_CATEGORY  // Collide only with walls, obstacles, and targets
    }
  });
  balls.push(ball);
  World.add(engine.world, [ball]);
}

// Initialize Q tables for each ball
balls.forEach((ball) => {
  Q.set(ball.id.toString(), new Map());
});

// Scores
let scores: {[id: string]: number} = {};

// Initialize score for each ball
balls.forEach((ball) => {
  scores[ball.id] = 0;
});

const obstacleOptions = {
  isStatic: true,
  collisionFilter: {
    category: OBSTACLE_CATEGORY,
    mask: BALL_CATEGORY,
  }
};

// Create and add obstacles to the world
const obstacles = [
  // Rectangular obstacles
  Bodies.rectangle(400, 300, 100, 20, obstacleOptions),
  Bodies.rectangle(600, 400, 80, 40, obstacleOptions),
  Bodies.rectangle(200, 150, 60, 30, obstacleOptions),

  // Circular obstacles
  Bodies.circle(300, 500, 30, obstacleOptions),
  Bodies.circle(700, 200, 20, obstacleOptions),

  // More rectangular obstacles
  Bodies.rectangle(500, 100, 120, 20, obstacleOptions),
  Bodies.rectangle(100, 500, 40, 60, obstacleOptions),

  // More circular obstacles
  Bodies.circle(100, 200, 15, obstacleOptions),
  Bodies.circle(600, 500, 25, obstacleOptions),

  // Additional obstacle shapes
  Bodies.trapezoid(400, 450, 60, 40, 0.7, obstacleOptions), // Trapezoid
  Bodies.polygon(700, 350, 5, 30, obstacleOptions), // Pentagon
  Bodies.fromVertices(300, 100, [[0, 0] as any, [40, 0], [20, 30]], obstacleOptions) // Custom polygon
];

World.add(engine.world, obstacles);

// Adding a target
const target = Bodies.circle(700, 100, 60, {
  isStatic: true,
  collisionFilter: {
    category: TARGET_CATEGORY,
    mask: BALL_CATEGORY
  }
});
World.add(engine.world, [target]);

// Helper Functions
const getState = (obj: any) => `${Math.floor(obj.position.x / 100)}:${Math.floor(obj.position.y / 100)}`;

const getExtendedState = (obj: any) => {
  const roundedX = Math.round(obj.position.x / 100);
  const roundedY = Math.round(obj.position.y / 100);
  const roundedVX = Math.sign(obj.velocity.x);
  const roundedVY = Math.sign(obj.velocity.y);
  return `${roundedX}:${roundedY}:${roundedVX}:${roundedVY}`;
};

const updateQValue = (s: string, a: string, r: number, sNext: string, ballId: string) => {
  const qTable = Q.get(ballId);

  if (!qTable) {
    console.log('no q table found!');
    return;
  }
  
  if (!qTable.has(`${s}:${a}`)) {
    qTable.set(`${s}:${a}`, Math.random()); // Initialize Q-value randomly
  }
  
  const currentQ = qTable.get(`${s}:${a}`)!;
  const maxQNext = Math.max(
    qTable.get(`${sNext}:left`) ?? Math.random(),
    qTable.get(`${sNext}:right`) ?? Math.random(),
    qTable.get(`${sNext}:up`) ?? Math.random(),
    qTable.get(`${sNext}:down`) ?? Math.random()
  );

  const updatedQ = currentQ + ALPHA * (r + GAMMA * maxQNext - currentQ);
  qTable.set(`${s}:${a}`, updatedQ);
};

const chooseAction = (state: string, ballId: string) => {
  const qTable = Q.get(ballId) ?? new Map();
  if (Math.random() < EPSILON) {
    // Exploration: random action
    const actions = ['left', 'right', 'up', 'down'];
    return actions[Math.floor(Math.random() * actions.length)];
  } else {
    // Exploitation: action with highest Q-value
    const qLeft = qTable.get(`${state}:left`) ?? Math.random();
    const qRight = qTable.get(`${state}:right`) ?? Math.random();
    const qUp = qTable.get(`${state}:up`) ?? Math.random();
    const qDown = qTable.get(`${state}:down`) ?? Math.random();
    
    const maxQ = Math.max(qLeft, qRight, qUp, qDown);
    
    let bestActions = [];
    if (qLeft === maxQ) bestActions.push('left');
    if (qRight === maxQ) bestActions.push('right');
    if (qUp === maxQ) bestActions.push('up');
    if (qDown === maxQ) bestActions.push('down');
    
    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }
};

// Define a function to reset rewards and update states
const updateRL = (ball: any, defaultReward: number, _index: number) => {
  if (successfulBall) {
    console.log('reset!');
    if (successfulBall.id !== ball.id) {
      // Clone the successful ball and mutate its Q-values
      const clonedBall = Bodies.circle(ballStartX, ballStartY, ballRadius, {
        collisionFilter: {
          category: BALL_CATEGORY,
          mask: WALL_CATEGORY | OBSTACLE_CATEGORY | TARGET_CATEGORY  // Collide only with walls, obstacles, and targets
        }
      });
      scores[clonedBall.id] = 0;
      // Mutate Q-values for the cloned ball (e.g., add small random values)
      Q.set(clonedBall.id.toString(), mutateQValues(Q.get(successfulBall.id.toString()) || new Map()));
      balls.splice(balls.findIndex(findBall => findBall.id === ball.id), 1, clonedBall);
      World.remove(engine.world, [ball]);
      World.add(engine.world, [clonedBall]);
    }
    scores[ball.id] = 0;
  }

  let reward = defaultReward;

  // Extended State and Reward
  const extendedCurrentState = getExtendedState(ball);

  if (successfulBall) {
    if (successfulBall.id !== ball.id) {
      reward = -0.5; // Penalty for not being the successful ball
    } else {
      reward = 1; // Reward for being the successful ball
    }
  }

  const action = chooseAction(extendedCurrentState, ball.id.toString());
  const nextState = getState(ball);
  updateQValue(extendedCurrentState, action, reward, nextState, ball.id.toString());
  
  if (scores[ball.id] > bestScore) {
    bestScore = scores[ball.id];
  }

  if (scores[ball.id] < worstScore) {
    worstScore = scores[ball.id];
  }

  if (action === 'left') {
    Body.setVelocity(ball, { x: -5, y: 0 });
  } else if (action === 'right') {
    Body.setVelocity(ball, { x: 5, y: 0 });
  } else if (action === 'up') {
    Body.setVelocity(ball, { x: 0, y: -5 });
  } else if (action === 'down') {
    Body.setVelocity(ball, { x: 0, y: 5 });
  }
};

Events.on(engine, 'afterUpdate', () => {
  if (episodeCount < MAX_EPISODES) {
    bestScore = -Infinity;
    worstScore = Infinity;

    balls.forEach((ball, index) => {
      // 0 reward for non-collision
      updateRL(ball, 0, index);
    });

    successfulBall = null;

    // Update the chart
    myChart.data.labels!.push(`${episodeCount}`);
    myChart.data.datasets![0].data!.push(bestScore);
    myChart.data.datasets![1].data!.push(worstScore);
    myChart.update();

    // console.log(`Episode ${episodeCount} - Best Score: ${bestScore} - Worst Score: ${worstScore}`);
    episodeCount++;
    document.getElementById('episodes')!.innerHTML = 'Episode ' + episodeCount.toString();
  } else {
    location.reload();
  }
});

// Define a variable to track the successful ball
let successfulBall: any | null = null;
let lastSuccessfulBall: any | null = null;
let successfulBallStreak: 0;
let previousEpisodeCount = 0;

// RL Loop
Events.on(engine, 'collisionStart', (event) => {
  const pairs = event.pairs;

  pairs.forEach((pair) => {
    balls.forEach((ball, _index) => {
      if (successfulBall) {
        return;
      }

      if (pair.bodyA === ball || pair.bodyB === ball) {
        const otherBody = pair.bodyA === ball ? pair.bodyB : pair.bodyA;
        let reward = 0;

        if (otherBody === target) {
          reward = 1;
          if (ball.id in scores) {
            scores[ball.id] += 1;
          }

          if (previousEpisodeCount) {
            updatePerformance(previousEpisodeCount - episodeCount);
          }

          previousEpisodeCount = episodeCount;

          console.log('success with ' + episodeCount + ' episodes!');
          console.log(Q);
          successfulBall = ball;
          iterationsCount++;
          document.getElementById('iterations')!.innerHTML = 'Iteration ' + iterationsCount.toString();
          if (lastSuccessfulBall === successfulBall) {
            successfulBallStreak++;
          } else {
            successfulBallStreak = 0;
          }
          lastSuccessfulBall = successfulBall;
          document.getElementById('successful')!.innerHTML = 'Last successful ball ID: ' + ball.id.toString();
          document.getElementById('streak')!.innerHTML = 'Streak (same ball): ' + successfulBallStreak.toString();
          document.getElementById('rate')!.innerHTML = 'Learning rate: ' + currentMutationAlpha.toPrecision(3);
          myChart.data.labels!.splice(0, myChart.data.labels!.length);
          myChart.data.datasets![0].data!.splice(0, myChart.data.datasets![0].data!.length);
          myChart.data.datasets![1].data!.splice(0, myChart.data.datasets![1].data!.length);
          myChart.update();
          iterationsChart.data.labels!.push(`${iterationsCount}`);
          iterationsChart.data.datasets![0].data!.push(episodeCount);
          iterationsChart.data.datasets![1].data!.push(currentMutationAlpha * 1000);
          iterationsChart.update();
          episodeCount = 0;

          // Reset the successful ball's position and velocity
          Body.setPosition(successfulBall, { x: ballStartX, y: ballStartY });
          Body.setVelocity(successfulBall, { x: 0, y: 0 });

          return;
        } else if (obstacles.includes(otherBody) || otherBody === wallTop || otherBody === wallBottom || otherBody === wallLeft || otherBody === wallRight) {
          reward = -1;
          if (ball.id in scores) {
            scores[ball.id] -= 1;
          }
        }

        // Update Q-values for the current ball
        const currentState = getState(ball);
        const action = chooseAction(currentState, ball.id);
        const nextState = getState(ball);
        updateQValue(currentState, action, reward, nextState, ball.id.toString());
      }
    });
  });
});

let lastPerformanceAverage: number | undefined;
let lastPerformances: number[] = [];

// Function to update last five performances and average
const updatePerformance = (newPerformance: number) => {
  if (lastPerformances.length >= RELATIVE_PERFORMANCE_AVERAGE_DURATION) {
    lastPerformances.shift();
  }
  lastPerformances.push(newPerformance);
  
  if (lastPerformances.length < RELATIVE_PERFORMANCE_AVERAGE_DURATION) {
    return;
  }
  
  const averagePerformance = lastPerformances.reduce((acc, val) => acc + val, 0) / lastPerformances.length;

  if (lastPerformanceAverage === undefined) {
    lastPerformanceAverage = averagePerformance;
    return;
  }

  if (averagePerformance <= lastPerformanceAverage * RELATIVE_PERFORMANCE_THRESHOLD) {
    console.log('apply mutation alpha correction!');
    currentMutationAlpha /= MUTATION_DECAY**2; // Increase alpha
    if (currentMutationAlpha > 0.5) {
      currentMutationAlpha = 0.5; // Set an upper bound
    }
  }

  lastPerformanceAverage = averagePerformance;
};

let currentMutationAlpha = INITIAL_MUTATION_ALPHA; // track the current mutation rate

// Existing mutateQValues function
const mutateQValues = (originalQTable: Map<string, number>) => {
  const mutatedQTable = new Map<string, number>();

  originalQTable.forEach((value, key) => {
    const randomFactor = Math.random() * 2 - 1;
    const mutatedValue = value + currentMutationAlpha * randomFactor;
    mutatedQTable.set(key, mutatedValue);
  });

  // Decay the mutation alpha
  currentMutationAlpha *= MUTATION_DECAY;
  if (currentMutationAlpha < 0.01) {
    currentMutationAlpha = 0.01; // set a lower bound
  }

  return mutatedQTable;
};

// Run everything
const runner = Runner.create();
Runner.run(runner, engine);
Render.run(render);