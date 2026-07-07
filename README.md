<img width="802" height="540" alt="Screenshot 2026-07-07 at 3 34 25 PM" src="https://github.com/user-attachments/assets/b11aec2a-2dc8-43bc-80ca-941de1ec4938" />
# Latent Space Knowledge Graph Embedding (KGE) Explorer

A **Knowledge Graph Embedding (KGE) Engine and Latent Space Explorer** running 100% locally in your browser. It allows users to train, visualize, and query embedding models on custom or preloaded demonstration datasets, perform vector arithmetic to generate synthetic entities, detect relational anomalies, and interpret latent structures using optional server-side LLMs or local models.

---

## Key Capabilities

*   **Four KGE Formulations**: Train **TransE**, **DistMult**, **ComplEx**, or **RotatE** models directly in the browser using custom learning rates, margins, dimensions, and negative sampling configurations.
*   **Dual-Projection 3D/2D Engine**:
    *   **PCA (Principal Component Analysis)** via Power Iteration with Deflation for high-dimensional coordinate alignment.
    *   **UMAP-style Force-Directed Projection** for non-linear, alignment-friendly relative distances.
    *   *Features full label toggling, rotational reset, and zoomable interactive viewport controls.*
*   **Vector Arithmetic**: Generate synthetic latent entities through algebraic combinations of existing entities (e.g., $e_{\text{new}} = e_A + e_B - e_C$) and perform real-time link prediction over their relational neighborhoods.
*   **Anomaly & Violation Detection**: Run out-of-distribution (OOD) tests using mathematical energy thresholds to discover inconsistent facts or noise in the graph.
*   **Dynamic Clustering**: High-dimensional **K-Means Clustering** directly over latent embedding spaces to discover modularity and semantic sub-communities.
*   **Optional LLM Interpretation**: Connect to **Gemini** (with automatic exponential backoff retry for transient high-demand limits) or point the dashboard to **local LLMs (Ollama)** to generate semantic synthesis and relationship analysis from latent vector patterns.

---

## Mathematical Formulations

Knowledge Graph Embeddings represent entities $e \in \mathcal{E}$ and relations $r \in \mathcal{R}$ as dense vectors (or complex representations) in a continuous low-dimensional vector space. The goal is to maximize a scoring function $f_r(h, t)$ for true triples $(h, r, t)$ and minimize it for corrupted negative samples.

### 1. KGE Score Functions $f_r(h, t)$

#### **TransE (Translational Embeddings)**
Models relations as translation vectors in real space $\mathbb{R}^d$:
$$f_r(h, t) = -\|h + r - t\|_2^2$$
*   *Interpretation*: If the relation holds, then the tail vector $t$ should be close to the head vector $h$ shifted by the relation vector $r$ ($h + r \approx t$).
*   *Constraints*: Entity embeddings are projected onto the unit ball ($|e\|_2 \le 1$) after each gradient step to prevent the optimization from trivial scaling.

#### **DistMult (Bilinear Diagonal)**
Models relations as diagonal matrices in real space $\mathbb{R}^d$:
$$f_r(h, t) = \langle h, r, t \rangle = \sum_{i=1}^{d} h_i \cdot r_i \cdot t_i$$
*   *Interpretation*: Captures symmetric pairwise interactions. It is a highly efficient baseline but limited to symmetric relational patterns due to commutative multiplication.

#### **ComplEx (Complex Bilinear)**
Extends entities and relations to the complex domain $\mathbb{C}^d$:
$$f_r(h, t) = \text{Re}(\langle h, r, \bar{t} \rangle) = \sum_{i=1}^{d} \text{Re}\left(h_i \cdot r_i \cdot \bar{t}_i\right)$$
where $\bar{t}$ is the complex conjugate of $t$. Expanding the real and imaginary parts ($e = e_{\text{Re}} + i e_{\text{Im}}$) yields:
$$f_r(h, t) = \sum_{i=1}^{d} \left(h_{i,\text{Re}}r_{i,\text{Re}}t_{i,\text{Re}} + h_{i,\text{Im}}r_{i,\text{Re}}t_{i,\text{Im}} + h_{i,\text{Re}}r_{i,\text{Im}}t_{i,\text{Im}} - h_{i,\text{Im}}r_{i,\text{Im}}t_{i,\text{Re}}\right)$$
*   *Interpretation*: The asymmetry of the complex conjugate product allows ComplEx to naturally model asymmetric relational relations (e.g., $A \xrightarrow{\text{parent}} B$ does not imply $B \xrightarrow{\text{parent}} A$).

#### **RotatE (Rotational Embeddings)**
Models relations as rotation operations in complex space $\mathbb{C}^d$:
$$f_r(h, t) = -\|h \circ r - t\|_2^2$$
where each component $r_i$ is constrained to lie on the unit circle: $r_i = e^{i\theta_i} = \cos(\theta_i) + i\sin(\theta_i)$.
*   *Interpretation*: The relation acts as a rotation of the head entity in the complex plane to align with the tail entity.
*   *Representational Power*: Mathematically proven to capture symmetry, asymmetry, inversion (e.g., *hypernym* vs. *hyponym*), and composition relational patterns.

---

### 2. High-Dimensional 3D Projection Algorithms

#### **A. Principal Component Analysis (PCA)**
To project high-dimensional coordinates (dimension $d$) down to visualizable 3-dimensional space:
1.  **Mean-Centering**: Center the dataset to have zero mean along all embedding dimensions.
2.  **Covariance Computation**: Construct the $d \times d$ covariance matrix $\Sigma = \frac{1}{N-1} X^T X$.
3.  **Power Iteration with Deflation**: Find the top 3 dominant eigenvectors $v_1, v_2, v_3$ iteratively:
    *   Start with a random unit vector $u_0$.
    *   Iterate $u_{k+1} = \frac{\Sigma u_k}{\|\Sigma u_k\|}$ until convergence to find $v_p$ and eigenvalue $\lambda_p$.
    *   **Deflate** the covariance matrix: $\Sigma \leftarrow \Sigma - \lambda_p v_p v_p^T$ and repeat to find the next orthogonal eigenvector.
4.  **Projection**: Project the original centered embeddings onto the subspace spanned by these top 3 eigenvectors.

#### **B. Non-Linear Neighborhood Force Projection (UMAP-inspired)**
For preserving topological neighborhoods and local clusters, the engine supports a non-linear force-directed layout initialized by PCA coordinates:
1.  **Distance Matrix**: Construct a pairwise high-dimensional Euclidean distance matrix $D_{ij} = \|e_i - e_j\|_2$ across high-dimensional embeddings.
2.  **K-Nearest Neighbor Graph (KNN)**: Build a directed neighborhood graph where each node is connected to its top $k$ closest neighbors ($k \approx 5$).
3.  **Attractive Force**: Pull connected neighbor nodes together with a force proportional to distance:
    $$F_{\text{attractive}}(u, v) = c_{\text{attr}} \cdot \alpha \cdot d(u, v)$$
4.  **Repulsive Force**: Push non-neighboring nodes apart using an inverse-quadratic force barrier to prevent overlapping:
    $$F_{\text{repulsive}}(u, v) = \frac{c_{\text{rep}} \cdot \alpha}{d(u, v)^2 + 1.0}$$
5.  **Dynamic Cooling**: Decay the learning rate factor $\alpha$ over $80$ iterations to freeze coordinates into a stable topological state.

---

## Local Development Setup

To run this application locally on your machine, follow these steps:

### Prerequisites
*   [Node.js](https://nodejs.org/) (v18 or higher recommended)
*   npm (installed automatically with Node.js)

### Installation
1.  Clone or download this repository.
2.  Open your terminal inside the root directory.
3.  Install the required dependencies:
    ```bash
    npm install
    ```

### Environment Setup
Create a `.env` file in the root directory to store your credentials (optional, only required if using Gemini API for semantic explanations):
```env
# Create a .env file
GEMINI_API_KEY=your_actual_google_gemini_api_key_here
```

### Run the App
Start the development server (which spins up both Express backend API and Vite UI proxy):
```bash
npm run dev
```
Open your browser and navigate to **`http://localhost:3000`**.

---

## Configuring Local LLM (Ollama)

If you prefer to run 100% offline or avoid cloud API keys, you can integrate your local LLM server using **Ollama**:

### 1. Download and Start Ollama
1.  Install Ollama from [ollama.com](https://ollama.com).
2.  Open your terminal and pull your preferred model (e.g., `gemma2`, `llama3`, or `mistral`):
    ```bash
    ollama pull gemma2
    ```
3.  Verify Ollama is running and listening on its default port:
    ```bash
    curl http://localhost:11434
    ```

### 2. Configure the Explorer UI
In the **Latent Space KGE Explorer** UI:
1.  Locate the **LLM & Cognitive Config** panel in the dashboard sidebar.
2.  Set the **Provider** dropdown to **Ollama**.
3.  Set the **Ollama Endpoint** to your local host (usually `http://localhost:11434` or `http://127.0.0.1:11434`).
4.  Provide the exact **Ollama Model** name you downloaded (e.g., `gemma2` or `llama3`).
5.  Click **Apply Config** to route all interpretation and latent explanation queries straight to your local hardware.
