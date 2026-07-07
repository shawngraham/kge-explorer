/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * High-performance Knowledge Graph Embedding (KGE) Engine in TypeScript.
 * Implements TransE, DistMult, ComplEx, and RotatE models with exact analytical gradients.
 */

export interface Triple {
  id: string;
  sub: string;
  rel: string;
  obj: string;
}

export type KGEModelType = 'transE' | 'distMult' | 'complEx' | 'rotatE';

export interface TrainingConfig {
  modelType: KGEModelType;
  dim: number;          // Embedding dimension
  lr: number;           // Learning rate
  margin: number;       // Margin for ranking loss
  epochs: number;
  negSamples: number;   // Number of negative samples per positive triple
}

export interface TrainingProgress {
  epoch: number;
  loss: number;
  entityEmbeddings: { [entity: string]: number[] }; // Projected 3D coordinates
  relationEmbeddings: { [relation: string]: number[] }; // Projected 3D coordinates
}

// Helper: Vector operations
const vecDot = (a: number[], b: number[]): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
};

const vecNorm = (a: number[]): number => {
  return Math.sqrt(vecDot(a, a));
};

const vecNormalize = (a: number[]): number[] => {
  const norm = vecNorm(a);
  if (norm < 1e-8) return a.map(() => 0);
  return a.map(v => v / norm);
};

// PCA (Power Iteration with Deflation) to project embeddings to 3D space
export function projectTo3D(embeddings: { [key: string]: number[] }): { [key: string]: number[] } {
  const keys = Object.keys(embeddings);
  if (keys.length === 0) return {};
  
  const dim = embeddings[keys[0]].length;
  
  // If dimension is already 3 or less, pad with zeros or return as is
  if (dim <= 3) {
    const projected: { [key: string]: number[] } = {};
    keys.forEach(k => {
      const vec = embeddings[k];
      projected[k] = [vec[0] || 0, vec[1] || 0, vec[2] || 0];
    });
    return projected;
  }

  // 1. Convert to matrix and mean-center
  const data = keys.map(k => [...embeddings[k]]);
  const numSamples = data.length;
  const means = new Array(dim).fill(0);
  
  for (let d = 0; d < dim; d++) {
    let sum = 0;
    for (let s = 0; s < numSamples; s++) {
      sum += data[s][d];
    }
    means[d] = sum / numSamples;
  }
  
  for (let s = 0; s < numSamples; s++) {
    for (let d = 0; d < dim; d++) {
      data[s][d] -= means[d];
    }
  }

  // 2. Compute Covariance Matrix (dim x dim)
  const cov = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      let sum = 0;
      for (let s = 0; s < numSamples; s++) {
        sum += data[s][i] * data[s][j];
      }
      cov[i][j] = sum / Math.max(1, numSamples - 1);
    }
  }

  // 3. Find top 3 eigenvectors using Power Iteration with Deflation
  const eigenvectors: number[][] = [];
  
  for (let pc = 0; pc < 3; pc++) {
    // Initial random vector
    let v = Array.from({ length: dim }, () => Math.random() - 0.5);
    v = vecNormalize(v);
    
    let eigenvalue = 0;
    const maxIter = 100;
    
    for (let iter = 0; iter < maxIter; iter++) {
      // Multiply cov * v
      const nextV = new Array(dim).fill(0);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          nextV[i] += cov[i][j] * v[j];
        }
      }
      
      const norm = vecNorm(nextV);
      if (norm < 1e-8) break;
      
      // Check convergence
      let diff = 0;
      for (let i = 0; i < dim; i++) {
        diff += Math.abs(v[i] - nextV[i] / norm);
      }
      
      v = nextV.map(x => x / norm);
      eigenvalue = norm;
      
      if (diff < 1e-6) break;
    }
    
    eigenvectors.push(v);
    
    // Deflate Covariance Matrix: cov = cov - eigenvalue * v * v^T
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        cov[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  // 4. Project data onto the top 3 eigenvectors
  const projected: { [key: string]: number[] } = {};
  keys.forEach((key, s) => {
    const sample = keys.map(k => embeddings[k])[s];
    // Subtract mean
    const centered = sample.map((val, d) => val - means[d]);
    
    // Project
    const x = vecDot(centered, eigenvectors[0]);
    const y = vecDot(centered, eigenvectors[1]);
    const z = vecDot(centered, eigenvectors[2]);
    projected[key] = [x, y, z];
  });

  return projected;
}

// UMAP-style Non-linear Neighborhood Projection
export function projectTo3DWithUMAP(embeddings: { [key: string]: number[] }): { [key: string]: number[] } {
  const keys = Object.keys(embeddings);
  if (keys.length === 0) return {};
  
  // 1. Start with PCA coordinates as stable initialization
  const pcaCoords = projectTo3D(embeddings);
  
  const coords: { [key: string]: number[] } = {};
  keys.forEach(k => {
    coords[k] = [...pcaCoords[k]];
  });
  
  const numNodes = keys.length;
  if (numNodes <= 3) return coords;

  const dim = embeddings[keys[0]].length;
  
  // 2. Compute high-dimensional distance matrix
  const distMatrix = Array.from({ length: numNodes }, () => new Float32Array(numNodes));
  for (let i = 0; i < numNodes; i++) {
    for (let j = i; j < numNodes; j++) {
      if (i === j) {
        distMatrix[i][j] = 0;
      } else {
        const v1 = embeddings[keys[i]];
        const v2 = embeddings[keys[j]];
        let sumSqr = 0;
        const len = Math.min(v1.length, v2.length);
        for (let d = 0; d < len; d++) {
          const diff = v1[d] - v2[d];
          sumSqr += diff * diff;
        }
        const d = Math.sqrt(sumSqr);
        distMatrix[i][j] = d;
        distMatrix[j][i] = d;
      }
    }
  }

  // 3. Find k-nearest neighbors (k = min(5, numNodes - 1))
  const kNeighbors = Math.min(5, numNodes - 1);
  const knn: Set<number>[] = Array.from({ length: numNodes }, () => new Set());
  
  for (let i = 0; i < numNodes; i++) {
    const sortedIndices = Array.from({ length: numNodes }, (_, idx) => idx)
      .filter(idx => idx !== i)
      .sort((a, b) => distMatrix[i][a] - distMatrix[i][b]);
    
    for (let neighborIdx = 0; neighborIdx < kNeighbors; neighborIdx++) {
      knn[i].add(sortedIndices[neighborIdx]);
    }
  }

  // 4. Force-directed optimization loop
  const iterations = 80;
  let alpha = 1.0; // learning rate decay
  
  const vel: { [key: string]: number[] } = {};
  keys.forEach(k => {
    vel[k] = [0, 0, 0];
  });

  const attractiveStrength = 0.08;
  const repulsiveStrength = 8.0;
  
  for (let iter = 0; iter < iterations; iter++) {
    const forces: { [key: string]: number[] } = {};
    keys.forEach(k => {
      forces[k] = [0, 0, 0];
    });

    // 4.1 Attractive forces
    for (let i = 0; i < numNodes; i++) {
      const uKey = keys[i];
      const uPos = coords[uKey];
      
      knn[i].forEach(j => {
        const vKey = keys[j];
        const vPos = coords[vKey];
        
        const dx = vPos[0] - uPos[0];
        const dy = vPos[1] - uPos[1];
        const dz = vPos[2] - uPos[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-3;
        
        const forceMag = attractiveStrength * alpha * dist;
        forces[uKey][0] += (dx / dist) * forceMag;
        forces[uKey][1] += (dy / dist) * forceMag;
        forces[uKey][2] += (dz / dist) * forceMag;
      });
    }

    // 4.2 Repulsive forces
    for (let i = 0; i < numNodes; i++) {
      const uKey = keys[i];
      const uPos = coords[uKey];
      
      for (let j = 0; j < numNodes; j++) {
        if (i === j) continue;
        const vKey = keys[j];
        const vPos = coords[vKey];
        
        const dx = uPos[0] - vPos[0];
        const dy = uPos[1] - vPos[1];
        const dz = uPos[2] - vPos[2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-3;
        
        const forceMag = (repulsiveStrength * alpha) / (dist * dist + 1.0);
        forces[uKey][0] += (dx / dist) * forceMag;
        forces[uKey][1] += (dy / dist) * forceMag;
        forces[uKey][2] += (dz / dist) * forceMag;
      }
    }

    // 4.3 Update positions
    for (let i = 0; i < numNodes; i++) {
      const k = keys[i];
      const f = forces[k];
      const v = vel[k];
      const p = coords[k];
      
      v[0] = v[0] * 0.82 + f[0];
      v[1] = v[1] * 0.82 + f[1];
      v[2] = v[2] * 0.82 + f[2];
      
      p[0] += v[0];
      p[1] += v[1];
      p[2] += v[2];
    }
    
    alpha *= 0.98;
  }

  return coords;
}

// K-Means Clustering on high-dimensional vectors
export function kmeansClustering(embeddings: { [key: string]: number[] }, kCount: number): { [key: string]: number } {
  const keys = Object.keys(embeddings);
  if (keys.length === 0) return {};
  
  const numNodes = keys.length;
  const K = Math.min(kCount, numNodes);
  if (K <= 1) {
    const result: { [key: string]: number } = {};
    keys.forEach(k => { result[k] = 0; });
    return result;
  }

  const dim = embeddings[keys[0]].length;
  
  // Initialize centroids
  const centroids: number[][] = [];
  const chosenIndices = new Set<number>();
  while (centroids.length < K) {
    const idx = Math.floor(Math.random() * numNodes);
    if (!chosenIndices.has(idx)) {
      chosenIndices.add(idx);
      centroids.push([...embeddings[keys[idx]]]);
    }
  }

  const assignments: { [key: string]: number } = {};
  keys.forEach(k => { assignments[k] = 0; });

  const maxIter = 15;
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assign to nearest
    keys.forEach(key => {
      const vec = embeddings[key];
      let minDist = Infinity;
      let bestCentroid = 0;

      for (let c = 0; c < K; c++) {
        const cent = centroids[c];
        let sumSqr = 0;
        for (let d = 0; d < dim; d++) {
          const diff = vec[d] - cent[d];
          sumSqr += diff * diff;
        }
        if (sumSqr < minDist) {
          minDist = sumSqr;
          bestCentroid = c;
        }
      }

      if (assignments[key] !== bestCentroid) {
        assignments[key] = bestCentroid;
        changed = true;
      }
    });

    if (!changed) break;

    // Update centroids
    const sums = Array.from({ length: K }, () => new Array(dim).fill(0));
    const counts = new Array(K).fill(0);

    keys.forEach(key => {
      const centIdx = assignments[key];
      const vec = embeddings[key];
      counts[centIdx]++;
      for (let d = 0; d < dim; d++) {
        sums[centIdx][d] += vec[d];
      }
    });

    for (let c = 0; c < K; c++) {
      if (counts[c] > 0) {
        for (let d = 0; d < dim; d++) {
          centroids[c][d] = sums[c][d] / counts[c];
        }
      }
    }
  }

  return assignments;
}

export class KGEEngine {
  entities: string[] = [];
  relations: string[] = [];
  triples: Triple[] = [];
  
  // Map of entity/relation to index
  entity2idx: { [key: string]: number } = {};
  relation2idx: { [key: string]: number } = {};
  
  // Model embeddings
  // We use Float32Array to represent flat embeddings: [num_entities * dimension]
  // For complex models (ComplEx, RotatE), the dimension parameter represents the effective math dimension.
  // We allocate appropriately (e.g., ComplEx has real and imaginary vectors, so double parameters).
  entityEmbeddings: Float32Array = new Float32Array(0);
  relationEmbeddings: Float32Array = new Float32Array(0);
  
  // Auxiliary rotation states for RotatE (relations are represented as phase angles theta)
  relationPhases: Float32Array = new Float32Array(0);

  constructor(triples: Triple[]) {
    this.triples = triples;
    
    // Extract unique entities and relations
    const entitySet = new Set<string>();
    const relationSet = new Set<string>();
    
    triples.forEach(t => {
      entitySet.add(t.sub);
      entitySet.add(t.obj);
      relationSet.add(t.rel);
    });
    
    this.entities = Array.from(entitySet);
    this.relations = Array.from(relationSet);
    
    this.entities.forEach((e, idx) => this.entity2idx[e] = idx);
    this.relations.forEach((r, idx) => this.relation2idx[r] = idx);
  }

  // Initialize embeddings with Xavier/Glorot or random uniforms
  initialize(modelType: KGEModelType, dim: number) {
    const numEnt = this.entities.length;
    const numRel = this.relations.length;

    // Determine parameter size based on model type
    let entParamSize = dim;
    let relParamSize = dim;
    
    if (modelType === 'complEx') {
      // Real + Imaginary parts for entities and relations
      entParamSize = dim * 2;
      relParamSize = dim * 2;
    } else if (modelType === 'rotatE') {
      // RotatE has complex entities (dim * 2) and relations as rotation angles (dim)
      entParamSize = dim * 2;
      relParamSize = dim; // We'll represent relation as phase angle theta in [0, 2pi]
    }

    this.entityEmbeddings = new Float32Array(numEnt * entParamSize);
    this.relationEmbeddings = new Float32Array(numRel * relParamSize);

    // Initializer scale
    const entScale = 6.0 / Math.sqrt(entParamSize);
    const relScale = 6.0 / Math.sqrt(relParamSize);

    // Random uniform initialization
    for (let i = 0; i < this.entityEmbeddings.length; i++) {
      this.entityEmbeddings[i] = (Math.random() * 2 - 1) * entScale;
    }
    
    for (let i = 0; i < this.relationEmbeddings.length; i++) {
      if (modelType === 'rotatE') {
        // RotatE phase angles initialized in [-pi, pi]
        this.relationEmbeddings[i] = (Math.random() * 2 - 1) * Math.PI;
      } else {
        this.relationEmbeddings[i] = (Math.random() * 2 - 1) * relScale;
      }
    }
  }

  // Get specific vector from the flat array
  public getEntityVector(idx: number, size: number): Float32Array {
    return this.entityEmbeddings.subarray(idx * size, (idx + 1) * size);
  }

  public getRelationVector(idx: number, size: number): Float32Array {
    return this.relationEmbeddings.subarray(idx * size, (idx + 1) * size);
  }

  // Exact scoring function for each model type
  scoreTriple(hIdx: number, rIdx: number, tIdx: number, modelType: KGEModelType, dim: number): number {
    if (modelType === 'transE') {
      const h = this.getEntityVector(hIdx, dim);
      const r = this.getRelationVector(rIdx, dim);
      const t = this.getEntityVector(tIdx, dim);
      
      // f_r(h, t) = -||h + r - t||_2^2
      let score = 0;
      for (let i = 0; i < dim; i++) {
        const diff = h[i] + r[i] - t[i];
        score -= diff * diff;
      }
      return score;
      
    } else if (modelType === 'distMult') {
      const h = this.getEntityVector(hIdx, dim);
      const r = this.getRelationVector(rIdx, dim);
      const t = this.getEntityVector(tIdx, dim);
      
      // f_r(h, t) = <h, r, t>
      let score = 0;
      for (let i = 0; i < dim; i++) {
        score += h[i] * r[i] * t[i];
      }
      return score;
      
    } else if (modelType === 'complEx') {
      const h = this.getEntityVector(hIdx, dim * 2);
      const r = this.getRelationVector(rIdx, dim * 2);
      const t = this.getEntityVector(tIdx, dim * 2);
      
      // ComplEx uses real and imaginary parts
      // Score = Re(<h, r, conj(t)>)
      let score = 0;
      for (let i = 0; i < dim; i++) {
        const h_re = h[i];
        const h_im = h[dim + i];
        const r_re = r[i];
        const r_im = r[dim + i];
        const t_re = t[i];
        const t_im = t[dim + i];
        
        score += (h_re * r_re * t_re) + (h_im * r_re * t_im) + (h_re * r_im * t_im) - (h_im * r_im * t_re);
      }
      return score;
      
    } else if (modelType === 'rotatE') {
      const h = this.getEntityVector(hIdx, dim * 2);
      const r_theta = this.getRelationVector(rIdx, dim); // represented as rotation phase
      const t = this.getEntityVector(tIdx, dim * 2);
      
      // f_r(h, t) = -||h * r - t||_2^2 where r_i = cos(theta_i) + i * sin(theta_i)
      let score = 0;
      for (let i = 0; i < dim; i++) {
        const h_re = h[i];
        const h_im = h[dim + i];
        const theta = r_theta[i];
        const t_re = t[i];
        const t_im = t[dim + i];
        
        const cos = Math.cos(theta);
        const sin = Math.sin(theta);
        
        // Rotated head: (h_re + i * h_im) * (cos + i * sin)
        const rot_re = h_re * cos - h_im * sin;
        const rot_im = h_re * sin + h_im * cos;
        
        const diff_re = rot_re - t_re;
        const diff_im = rot_im - t_im;
        
        score -= (diff_re * diff_re + diff_im * diff_im);
      }
      return score;
    }
    return 0;
  }

  // Normalize entity vectors to unit sphere (essential for TransE/RotatE stability)
  normalizeEntities(modelType: KGEModelType, dim: number) {
    const numEnt = this.entities.length;
    const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    
    for (let e = 0; e < numEnt; e++) {
      const start = e * size;
      let sumSq = 0;
      for (let i = 0; i < size; i++) {
        sumSq += this.entityEmbeddings[start + i] * this.entityEmbeddings[start + i];
      }
      const norm = Math.sqrt(sumSq);
      if (norm > 1.0) { // Limit embeddings to unit ball
        for (let i = 0; i < size; i++) {
          this.entityEmbeddings[start + i] /= norm;
        }
      }
    }
  }

  // Normalize relation vectors to unit sphere (keeps bilinear and translation models stable)
  normalizeRelations(modelType: KGEModelType, dim: number) {
    const numRel = this.relations.length;
    const size = modelType === 'complEx' ? dim * 2 : dim;
    
    for (let r = 0; r < numRel; r++) {
      const start = r * size;
      let sumSq = 0;
      for (let i = 0; i < size; i++) {
        sumSq += this.relationEmbeddings[start + i] * this.relationEmbeddings[start + i];
      }
      const norm = Math.sqrt(sumSq);
      if (norm > 1.0) { // Limit embeddings to unit ball
        for (let i = 0; i < size; i++) {
          this.relationEmbeddings[start + i] /= norm;
        }
      }
    }
  }

  // Train one epoch. Returns average epoch loss.
  trainEpoch(config: TrainingConfig): number {
    const { modelType, dim, lr, margin, negSamples } = config;
    const numEnt = this.entities.length;
    const sizeEnt = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    const sizeRel = modelType === 'complEx' ? dim * 2 : dim;
    
    let totalLoss = 0;
    let updatesCount = 0;
    let evaluationsCount = 0;

    // Shuffle triples
    const shuffledTriples = [...this.triples];
    for (let i = shuffledTriples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffledTriples[i], shuffledTriples[j]] = [shuffledTriples[j], shuffledTriples[i]];
    }

    shuffledTriples.forEach(triple => {
      const hIdx = this.entity2idx[triple.sub];
      const rIdx = this.relation2idx[triple.rel];
      const tIdx = this.entity2idx[triple.obj];
      
      if (hIdx === undefined || rIdx === undefined || tIdx === undefined) return;

      // Positive triple score
      const posScore = this.scoreTriple(hIdx, rIdx, tIdx, modelType, dim);

      // Generate negative samples (corrupt head or tail)
      for (let ns = 0; ns < negSamples; ns++) {
        let negHIdx = hIdx;
        let negTIdx = tIdx;
        
        // Randomly choose to corrupt head or tail
        if (Math.random() < 0.5) {
          negHIdx = Math.floor(Math.random() * numEnt);
          // Make sure it's not the same entity
          while (negHIdx === hIdx && numEnt > 1) {
            negHIdx = Math.floor(Math.random() * numEnt);
          }
        } else {
          negTIdx = Math.floor(Math.random() * numEnt);
          while (negTIdx === tIdx && numEnt > 1) {
            negTIdx = Math.floor(Math.random() * numEnt);
          }
        }

        const negScore = this.scoreTriple(negHIdx, rIdx, negTIdx, modelType, dim);

        // Margin ranking loss: Max(0, margin - posScore + negScore)
        // Since score is negated distance in TransE/RotatE, positive scores are closer to 0 (higher).
        const loss = Math.max(0, margin - posScore + negScore);
        
        evaluationsCount++;
        
        if (loss > 0) {
          totalLoss += loss;
          updatesCount++;

          // --- Backpropagation / Analytical Gradients ---
          
          if (modelType === 'transE') {
            const h = this.getEntityVector(hIdx, dim);
            const r = this.getRelationVector(rIdx, dim);
            const t = this.getEntityVector(tIdx, dim);
            
            const nh = this.getEntityVector(negHIdx, dim);
            const nt = this.getEntityVector(negTIdx, dim);

            // Gents for pos: d_pos = h + r - t
            // Gents for neg: d_neg = nh + r - nt
            for (let i = 0; i < dim; i++) {
              const d_pos = h[i] + r[i] - t[i];
              const d_neg = nh[i] + r[i] - nt[i];
              
              // Pos gradients (Loss minimizes d_pos^2, so gradient wrt posScore is -1.
              // Derivative of -Score_pos is 2 * d_pos. So gradient updates subtraction):
              // h_grad = 2 * d_pos
              // r_grad = 2 * d_pos
              // t_grad = -2 * d_pos
              h[i] -= lr * (2 * d_pos);
              r[i] -= lr * (2 * d_pos);
              t[i] += lr * (2 * d_pos);

              // Neg gradients (Loss maximizes d_neg^2, so gradient is opposite):
              // nh_grad = -2 * d_neg
              // r_grad = -2 * d_neg
              // nt_grad = 2 * d_neg
              nh[i] += lr * (2 * d_neg);
              r[i] += lr * (2 * d_neg);
              nt[i] -= lr * (2 * d_neg);
            }
            
          } else if (modelType === 'distMult') {
            const h = this.getEntityVector(hIdx, dim);
            const r = this.getRelationVector(rIdx, dim);
            const t = this.getEntityVector(tIdx, dim);
            
            const nh = this.getEntityVector(negHIdx, dim);
            const nt = this.getEntityVector(negTIdx, dim);

            for (let i = 0; i < dim; i++) {
              const h_val = h[i];
              const r_val = r[i];
              const t_val = t[i];
              const nh_val = nh[i];
              const nt_val = nt[i];

              // pos score = h * r * t
              // Gradient wrt h: -r*t, wrt r: -h*t, wrt t: -h*r
              h[i] += lr * (r_val * t_val);
              r[i] += lr * (h_val * t_val);
              t[i] += lr * (h_val * r_val);

              // neg score = nh * r * nt
              // Gradient wrt nh: r*nt, wrt r: nh*nt, wrt nt: nh*r
              nh[i] -= lr * (r_val * nt_val);
              r[i] -= lr * (nh_val * nt_val);
              nt[i] -= lr * (nh_val * r_val);
            }
            
          } else if (modelType === 'complEx') {
            const h = this.getEntityVector(hIdx, dim * 2);
            const r = this.getRelationVector(rIdx, dim * 2);
            const t = this.getEntityVector(tIdx, dim * 2);
            
            const nh = this.getEntityVector(negHIdx, dim * 2);
            const nt = this.getEntityVector(negTIdx, dim * 2);

            for (let i = 0; i < dim; i++) {
              const h_re = h[i];
              const h_im = h[dim + i];
              const r_re = r[i];
              const r_im = r[dim + i];
              const t_re = t[i];
              const t_im = t[dim + i];

              const nh_re = nh[i];
              const nh_im = nh[dim + i];
              const nt_re = nt[i];
              const nt_im = nt[dim + i];

              // Positive triple gradients
              const g_h_re = r_re * t_re + r_im * t_im;
              const g_h_im = r_re * t_im - r_im * t_re;
              const g_r_re = h_re * t_re + h_im * t_im;
              const g_r_im = h_re * t_im - h_im * t_re;
              const g_t_re = h_re * r_re - h_im * r_im;
              const g_t_im = h_im * r_re + h_re * r_im;

              h[i] += lr * g_h_re;
              h[dim + i] += lr * g_h_im;
              r[i] += lr * g_r_re;
              r[dim + i] += lr * g_r_im;
              t[i] += lr * g_t_re;
              t[dim + i] += lr * g_t_im;

              // Negative triple gradients
              const gn_h_re = r_re * nt_re + r_im * nt_im;
              const gn_h_im = r_re * nt_im - r_im * nt_re;
              const gn_r_re = nh_re * nt_re + nh_im * nt_im;
              const gn_r_im = nh_re * nt_im - nh_im * nt_re;
              const gn_t_re = nh_re * r_re - nh_im * r_im;
              const gn_t_im = nh_im * r_re + nh_re * r_im;

              nh[i] -= lr * gn_h_re;
              nh[dim + i] -= lr * gn_h_im;
              r[i] -= lr * gn_r_re;
              r[dim + i] -= lr * gn_r_im;
              nt[i] -= lr * gn_t_re;
              nt[dim + i] -= lr * gn_t_im;
            }
            
          } else if (modelType === 'rotatE') {
            const h = this.getEntityVector(hIdx, dim * 2);
            const r_theta = this.getRelationVector(rIdx, dim);
            const t = this.getEntityVector(tIdx, dim * 2);
            
            const nh = this.getEntityVector(negHIdx, dim * 2);
            const nt = this.getEntityVector(negTIdx, dim * 2);

            for (let i = 0; i < dim; i++) {
              const theta = r_theta[i];
              const cos = Math.cos(theta);
              const sin = Math.sin(theta);

              // Pos error
              const h_re = h[i];
              const h_im = h[dim + i];
              const t_re = t[i];
              const t_im = t[dim + i];
              const rot_re = h_re * cos - h_im * sin;
              const rot_im = h_re * sin + h_im * cos;
              const err_re = rot_re - t_re;
              const err_im = rot_im - t_im;

              // Update pos elements (loss decreases pos error)
              h[i] -= lr * (2 * (err_re * cos + err_im * sin));
              h[dim + i] -= lr * (2 * (-err_re * sin + err_im * cos));
              r_theta[i] -= lr * (2 * (err_re * (-h_re * sin - h_im * cos) + err_im * (h_re * cos - h_im * sin)));
              t[i] += lr * (2 * err_re);
              t[dim + i] += lr * (2 * err_im);

              // Neg error
              const nh_re = nh[i];
              const nh_im = nh[dim + i];
              const nt_re = nt[i];
              const nt_im = nt[dim + i];
              const nrot_re = nh_re * cos - nh_im * sin;
              const nrot_im = nh_re * sin + nh_im * cos;
              const nerr_re = nrot_re - nt_re;
              const nerr_im = nrot_im - nt_im;

              // Update neg elements (loss increases neg error)
              nh[i] += lr * (2 * (nerr_re * cos + nerr_im * sin));
              nh[dim + i] += lr * (2 * (-nerr_re * sin + nerr_im * cos));
              r_theta[i] += lr * (2 * (nerr_re * (-nh_re * sin - nh_im * cos) + nerr_im * (nh_re * cos - nh_im * sin)));
              nt[i] -= lr * (2 * nerr_re);
              nt[dim + i] -= lr * (2 * nerr_im);

              // Wrap angles to [-pi, pi]
              if (r_theta[i] > Math.PI) r_theta[i] -= 2 * Math.PI;
              if (r_theta[i] < -Math.PI) r_theta[i] += 2 * Math.PI;
            }
          }
        }
      }
    });

    // Enforce constraints (bounds on embeddings)
    this.normalizeEntities(modelType, dim);
    this.normalizeRelations(modelType, dim);

    return evaluationsCount > 0 ? totalLoss / evaluationsCount : 0;
  }

  // Retrieve current high-dimensional entity embeddings as coordinate dictionary
  getEntityEmbeddingsDict(modelType: KGEModelType, dim: number): { [entity: string]: number[] } {
    const dict: { [entity: string]: number[] } = {};
    const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    
    this.entities.forEach((entity, idx) => {
      const vec = this.getEntityVector(idx, size);
      dict[entity] = Array.from(vec);
    });
    
    return dict;
  }

  // Retrieve current high-dimensional relation embeddings as coordinate dictionary
  getRelationEmbeddingsDict(modelType: KGEModelType, dim: number): { [relation: string]: number[] } {
    const dict: { [relation: string]: number[] } = {};
    const size = modelType === 'complEx' ? dim * 2 : dim;
    
    this.relations.forEach((relation, idx) => {
      const vec = this.getRelationVector(idx, size);
      dict[relation] = Array.from(vec);
    });
    
    return dict;
  }

  // Calculate high-quality evaluation metrics: MRR, Hits@1, Hits5, Hits@10
  evaluateMetrics(modelType: KGEModelType, dim: number): { mrr: number; hits1: number; hits5: number; hits10: number } {
    if (this.triples.length === 0) {
      return { mrr: 0, hits1: 0, hits5: 0, hits10: 0 };
    }

    let sumReciprocalRank = 0;
    let countHits1 = 0;
    let countHits5 = 0;
    let countHits10 = 0;
    let totalEvals = 0;

    this.triples.forEach(triple => {
      const hIdx = this.entity2idx[triple.sub];
      const rIdx = this.relation2idx[triple.rel];
      const tIdx = this.entity2idx[triple.obj];
      if (hIdx === undefined || rIdx === undefined || tIdx === undefined) return;

      // 1. Tail corruption (h, r, ?)
      let tailRank = 1;
      const trueTailScore = this.scoreTriple(hIdx, rIdx, tIdx, modelType, dim);
      for (let entIdx = 0; entIdx < this.entities.length; entIdx++) {
        if (entIdx === tIdx) continue;
        const score = this.scoreTriple(hIdx, rIdx, entIdx, modelType, dim);
        if (score > trueTailScore) {
          tailRank++;
        }
      }
      sumReciprocalRank += 1.0 / tailRank;
      if (tailRank <= 1) countHits1++;
      if (tailRank <= 5) countHits5++;
      if (tailRank <= 10) countHits10++;
      totalEvals++;

      // 2. Head corruption (?, r, t)
      let headRank = 1;
      const trueHeadScore = trueTailScore;
      for (let entIdx = 0; entIdx < this.entities.length; entIdx++) {
        if (entIdx === hIdx) continue;
        const score = this.scoreTriple(entIdx, rIdx, tIdx, modelType, dim);
        if (score > trueHeadScore) {
          headRank++;
        }
      }
      sumReciprocalRank += 1.0 / headRank;
      if (headRank <= 1) countHits1++;
      if (headRank <= 5) countHits5++;
      if (headRank <= 10) countHits10++;
      totalEvals++;
    });

    if (totalEvals === 0) {
      return { mrr: 0, hits1: 0, hits5: 0, hits10: 0 };
    }

    return {
      mrr: sumReciprocalRank / totalEvals,
      hits1: countHits1 / totalEvals,
      hits5: countHits5 / totalEvals,
      hits10: countHits10 / totalEvals
    };
  }

  // Run a complete training run
  static train(
    triples: Triple[],
    config: TrainingConfig,
    onProgress: (progress: TrainingProgress) => void
  ): KGEEngine {
    const engine = new KGEEngine(triples);
    engine.initialize(config.modelType, config.dim);
    
    // We can run synchronous training in steps so it can report progress iteratively
    let currentEpoch = 0;
    
    const trainStep = () => {
      if (currentEpoch >= config.epochs) return;
      
      const loss = engine.trainEpoch(config);
      currentEpoch++;
      
      // Compute 3D projected coordinates for entities and relations
      const fullEntEmbed = engine.getEntityEmbeddingsDict(config.modelType, config.dim);
      const fullRelEmbed = engine.getRelationEmbeddingsDict(config.modelType, config.dim);
      
      const projectedEnt = projectTo3D(fullEntEmbed);
      const projectedRel = projectTo3D(fullRelEmbed);

      onProgress({
        epoch: currentEpoch,
        loss,
        entityEmbeddings: projectedEnt,
        relationEmbeddings: projectedRel,
      });

      if (currentEpoch < config.epochs) {
        // Queue next step
        setTimeout(trainStep, 0);
      }
    };

    trainStep();
    return engine;
  }

  // Core discovery features: Link Prediction
  predictLinks(
    subQuery: string | null,
    relQuery: string | null,
    objQuery: string | null,
    modelType: KGEModelType,
    dim: number,
    limit = 10
  ): { target: string; score: number; probability: number }[] {
    const results: { target: string; score: number; probability: number }[] = [];
    
    const rIdx = relQuery ? this.relation2idx[relQuery] : -1;
    const hIdx = subQuery ? this.entity2idx[subQuery] : -1;
    const tIdx = objQuery ? this.entity2idx[objQuery] : -1;

    // We can do predicting (h, r, ?)
    if (subQuery && relQuery && !objQuery) {
      if (hIdx === -1 || rIdx === -1) return [];
      
      this.entities.forEach((ent, entIdx) => {
        const score = this.scoreTriple(hIdx, rIdx, entIdx, modelType, dim);
        results.push({ target: ent, score, probability: 0 });
      });
      
    } 
    // We can do predicting (?, r, t)
    else if (!subQuery && relQuery && objQuery) {
      if (tIdx === -1 || rIdx === -1) return [];
      
      this.entities.forEach((ent, entIdx) => {
        const score = this.scoreTriple(entIdx, rIdx, tIdx, modelType, dim);
        results.push({ target: ent, score, probability: 0 });
      });
    }
    // We can do predicting (h, ?, t)
    else if (subQuery && !relQuery && objQuery) {
      if (hIdx === -1 || tIdx === -1) return [];
      
      this.relations.forEach((rel, r_index) => {
        const score = this.scoreTriple(hIdx, r_index, tIdx, modelType, dim);
        results.push({ target: rel, score, probability: 0 });
      });
    }

    // Sort descending by score (highest scores represent strongest link predictions)
    results.sort((a, b) => b.score - a.score);

    // Apply Softmax-like or min-max normalization to derive pseudoprobability percentages
    if (results.length > 0) {
      const scores = results.map(r => r.score);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const range = maxScore - minScore || 1.0;
      
      results.forEach(r => {
        r.probability = (r.score - minScore) / range;
      });
    }

    return results.slice(0, limit);
  }

  // Nearest semantic neighbors for entities (to explore structural neighborhoods)
  getNearestNeighbors(
    entity: string,
    modelType: KGEModelType,
    dim: number,
    limit = 8
  ): { entity: string; similarity: number }[] {
    const idx = this.entity2idx[entity];
    if (idx === undefined) return [];
    
    const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    const targetVec = Array.from(this.getEntityVector(idx, size));
    
    const results: { entity: string; similarity: number }[] = [];
    
    this.entities.forEach((ent, entIdx) => {
      if (ent === entity) return;
      const otherVec = Array.from(this.getEntityVector(entIdx, size));
      
      // Compute cosine similarity
      const dot = vecDot(targetVec, otherVec);
      const normA = vecNorm(targetVec);
      const normB = vecNorm(otherVec);
      const sim = normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
      
      results.push({ entity: ent, similarity: sim });
    });
    
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, limit);
  }

  // Discovery of latent structures: Find unconfirmed relationships with high similarity/score
  discoverLatentRelations(
    modelType: KGEModelType,
    dim: number,
    limit = 15
  ): { sub: string; rel: string; obj: string; score: number; confidence: number }[] {
    // Create a lookup map of actual training triples
    const actualSet = new Set(this.triples.map(t => `${t.sub}||${t.rel}||${t.obj}`));
    
    const candidates: { sub: string; rel: string; obj: string; score: number; confidence: number }[] = [];
    const entityCount = this.entities.length;
    const relationCount = this.relations.length;

    // For performance on large graphs, sample hypothetical triples
    const samplingSize = Math.min(3000, entityCount * relationCount * entityCount);
    
    for (let s = 0; s < samplingSize; s++) {
      const hIdx = Math.floor(Math.random() * entityCount);
      const rIdx = Math.floor(Math.random() * relationCount);
      const tIdx = Math.floor(Math.random() * entityCount);
      
      if (hIdx === tIdx) continue; // Skip self loops for discovery
      
      const sub = this.entities[hIdx];
      const rel = this.relations[rIdx];
      const obj = this.entities[tIdx];
      
      const key = `${sub}||${rel}||${obj}`;
      if (actualSet.has(key)) continue; // Must be unconfirmed (NOT in training data)
      
      const score = this.scoreTriple(hIdx, rIdx, tIdx, modelType, dim);
      candidates.push({ sub, rel, obj, score, confidence: 0 });
    }

    // Sort by highest score first
    candidates.sort((a, b) => b.score - a.score);
    const topCandidates = candidates.slice(0, limit);

    // Normalize confidence to 0-1
    if (topCandidates.length > 0) {
      const scores = topCandidates.map(c => c.score);
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);
      const range = maxScore - minScore || 1.0;
      
      topCandidates.forEach(c => {
        c.confidence = (c.score - minScore) / range;
      });
    }

    return topCandidates;
  }

  // Create a new synthetic latent entity based on vector arithmetic or centroid combinations
  generateLatentEntity(
    name: string,
    sourceEntities: string[],
    arithmeticOps: ('add' | 'sub')[],
    modelType: KGEModelType,
    dim: number
  ): { [rel: string]: { target: string; score: number }[] } {
    const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    const newVec = new Float32Array(size);
    
    if (sourceEntities.length === 0) return {};

    // Base entity
    const baseIdx = this.entity2idx[sourceEntities[0]];
    if (baseIdx === undefined) return {};
    const baseVec = this.getEntityVector(baseIdx, size);
    for (let i = 0; i < size; i++) newVec[i] = baseVec[i];

    // Apply vector operations: e.g. h1 + h2 - h3
    for (let i = 1; i < sourceEntities.length; i++) {
      const idx = this.entity2idx[sourceEntities[i]];
      if (idx === undefined) continue;
      const vec = this.getEntityVector(idx, size);
      
      const op = arithmeticOps[i - 1] || 'add';
      for (let d = 0; d < size; d++) {
        if (op === 'add') {
          newVec[d] += vec[d];
        } else {
          newVec[d] -= vec[d];
        }
      }
    }

    // Predict relationships for this new synthetic entity
    const predictions: { [rel: string]: { target: string; score: number }[] } = {};
    const numRelations = this.relations.length;

    // Temporarily add synthetic vector to embeddings representation
    // To score we can write a simple custom scorer that uses the new head vector directly
    const scoreWithCustomHead = (rIdx: number, tIdx: number): number => {
      if (modelType === 'transE') {
        const r = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim);
        let score = 0;
        for (let i = 0; i < dim; i++) {
          const diff = newVec[i] + r[i] - t[i];
          score -= diff * diff;
        }
        return score;
      } else if (modelType === 'distMult') {
        const r = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim);
        let score = 0;
        for (let i = 0; i < dim; i++) {
          score += newVec[i] * r[i] * t[i];
        }
        return score;
      } else if (modelType === 'complEx') {
        const r = this.getRelationVector(rIdx, dim * 2);
        const t = this.getEntityVector(tIdx, dim * 2);
        let score = 0;
        for (let i = 0; i < dim; i++) {
          const h_re = newVec[i];
          const h_im = newVec[dim + i];
          const r_re = r[i];
          const r_im = r[dim + i];
          const t_re = t[i];
          const t_im = t[dim + i];
          score += (h_re * r_re * t_re) + (h_im * r_re * t_im) + (h_re * r_im * t_im) - (h_im * r_im * t_re);
        }
        return score;
      } else if (modelType === 'rotatE') {
        const r_theta = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim * 2);
        let score = 0;
        for (let i = 0; i < dim; i++) {
          const h_re = newVec[i];
          const h_im = newVec[dim + i];
          const theta = r_theta[i];
          const t_re = t[i];
          const t_im = t[dim + i];
          const cos = Math.cos(theta);
          const sin = Math.sin(theta);
          const rot_re = h_re * cos - h_im * sin;
          const rot_im = h_re * sin + h_im * cos;
          const diff_re = rot_re - t_re;
          const diff_im = rot_im - t_im;
          score -= (diff_re * diff_re + diff_im * diff_im);
        }
        return score;
      }
      return 0;
    };

    // For each relation, rank the top objects
    this.relations.forEach((rel, rIdx) => {
      const relPredictions: { target: string; score: number }[] = [];
      this.entities.forEach((ent, tIdx) => {
        const score = scoreWithCustomHead(rIdx, tIdx);
        relPredictions.push({ target: ent, score });
      });
      predictions[rel] = relPredictions.sort((a, b) => b.score - a.score).slice(0, 5);
    });

    return predictions;
  }

  // Out-of-Distribution (OOD) Anomaly Detection using triple violation distances
  detectAnomalies(modelType: KGEModelType, dim: number): {
    triple: Triple;
    distance: number;
    zScore: number;
    isAnomaly: boolean;
  }[] {
    const anomalies: { triple: Triple; distance: number; zScore: number; isAnomaly: boolean }[] = [];
    if (this.triples.length === 0) return [];

    let totalDistSum = 0;
    const distances: number[] = [];

    this.triples.forEach(triple => {
      const hIdx = this.entity2idx[triple.sub];
      const rIdx = this.relation2idx[triple.rel];
      const tIdx = this.entity2idx[triple.obj];

      if (hIdx === undefined || rIdx === undefined || tIdx === undefined) {
        distances.push(0);
        return;
      }

      let dist = 0;

      if (modelType === 'transE') {
        const h = this.getEntityVector(hIdx, dim);
        const r = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim);
        let sumSq = 0;
        for (let i = 0; i < dim; i++) {
          const diff = h[i] + r[i] - t[i];
          sumSq += diff * diff;
        }
        dist = Math.sqrt(sumSq);
      } else if (modelType === 'distMult') {
        const h = this.getEntityVector(hIdx, dim);
        const r = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim);
        let sumSq = 0;
        for (let i = 0; i < dim; i++) {
          const diff = h[i] * r[i] - t[i];
          sumSq += diff * diff;
        }
        dist = Math.sqrt(sumSq);
      } else if (modelType === 'complEx') {
        const h = this.getEntityVector(hIdx, dim * 2);
        const r = this.getRelationVector(rIdx, dim * 2);
        const t = this.getEntityVector(tIdx, dim * 2);
        let sumSq = 0;
        for (let i = 0; i < dim; i++) {
          const h_re = h[i];
          const h_im = h[dim + i];
          const r_re = r[i];
          const r_im = r[dim + i];
          const t_re = t[i];
          const t_im = t[dim + i];

          const rot_re = h_re * r_re - h_im * r_im;
          const rot_im = h_re * r_im + h_im * r_re;
          const diff_re = rot_re - t_re;
          const diff_im = rot_im - t_im;
          sumSq += (diff_re * diff_re + diff_im * diff_im);
        }
        dist = Math.sqrt(sumSq);
      } else if (modelType === 'rotatE') {
        const h = this.getEntityVector(hIdx, dim * 2);
        const r_theta = this.getRelationVector(rIdx, dim);
        const t = this.getEntityVector(tIdx, dim * 2);
        let sumSq = 0;
        for (let i = 0; i < dim; i++) {
          const h_re = h[i];
          const h_im = h[dim + i];
          const theta = r_theta[i];
          const t_re = t[i];
          const t_im = t[dim + i];

          const cos = Math.cos(theta);
          const sin = Math.sin(theta);
          const rot_re = h_re * cos - h_im * sin;
          const rot_im = h_re * sin + h_im * cos;
          const diff_re = rot_re - t_re;
          const diff_im = rot_im - t_im;
          sumSq += (diff_re * diff_re + diff_im * diff_im);
        }
        dist = Math.sqrt(sumSq);
      }

      distances.push(dist);
      totalDistSum += dist;
    });

    const meanDist = totalDistSum / this.triples.length;
    let sumSqDiff = 0;
    distances.forEach(d => {
      sumSqDiff += (d - meanDist) * (d - meanDist);
    });
    const stdDev = Math.sqrt(sumSqDiff / this.triples.length) || 0.001;

    // Flag as anomaly if the distance is greater than mean + 1.2 * stdDev
    const threshold = meanDist + 1.2 * stdDev;

    this.triples.forEach((triple, idx) => {
      const dist = distances[idx];
      const zScore = (dist - meanDist) / stdDev;
      const isAnomaly = dist > threshold && zScore > 1.0;
      anomalies.push({
        triple,
        distance: dist,
        zScore,
        isAnomaly
      });
    });

    // Sort by descending distance/zScore to show worst violators first
    return anomalies.sort((a, b) => b.distance - a.distance);
  }
}
