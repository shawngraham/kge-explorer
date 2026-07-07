/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Full-stack Express server integrating Vite development middleware and Gemini API routes.
 */

import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;

// JSON parser
app.use(express.json());

// Lazy-initialized GoogleGenAI client to prevent startup crashes if key is missing
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error('GEMINI_API_KEY environment variable is required but missing. Configure it in Settings > Secrets.');
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

// ==========================================
// API Routes (Keep keys safe on the server!)
// ==========================================

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// 1. Interpret Relationship
async function generateLLMResponse(prompt: string, systemInstruction: string, llmConfig?: any): Promise<string> {
  const provider = llmConfig?.provider || 'gemini';
  
  if (provider === 'gemini') {
    const customKey = llmConfig?.geminiApiKey;
    const key = customKey || process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error('Gemini API key is not configured. Please set GEMINI_API_KEY in secrets or configure a custom API key in the settings panel.');
    }
    const ai = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const maxRetries = 3;
    let delayMs = 1500;
    let lastError: any = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await ai.models.generateContent({
          model: 'gemini-3.5-flash',
          contents: prompt,
          config: {
            systemInstruction,
            temperature: 0.7,
          },
        });
        return response.text || '';
      } catch (error: any) {
        lastError = error;
        const errMessage = error.message || '';
        const isTransient = error.status === 503 || error.status === 429 ||
                            errMessage.includes('503') || errMessage.includes('429') ||
                            errMessage.includes('UNAVAILABLE') || errMessage.includes('high demand') ||
                            errMessage.includes('temporarily');
        
        if (isTransient && attempt < maxRetries) {
          console.warn(`Gemini API returned transient error (attempt ${attempt}/${maxRetries}). Retrying in ${delayMs}ms... Error:`, errMessage);
          await new Promise(resolve => setTimeout(resolve, delayMs));
          delayMs *= 2.5; // Exponential backoff with a higher multiplier
        } else {
          break;
        }
      }
    }
    throw lastError;
  } else if (provider === 'ollama') {
    const endpoint = llmConfig?.ollamaEndpoint || 'http://localhost:11434';
    const model = llmConfig?.ollamaModel || 'gemma2';
    
    const cleanEndpoint = endpoint.replace(/\/$/, '');
    const url = `${cleanEndpoint}/api/chat`;
    
    const messages = [];
    if (systemInstruction) {
      messages.push({ role: 'system', content: systemInstruction });
    }
    messages.push({ role: 'user', content: prompt });
    
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        stream: false,
        options: {
          temperature: 0.7,
        }
      })
    });
    
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Ollama request failed: ${res.statusText}. ${errText}`);
    }
    
    const data: any = await res.json();
    return data.message?.content || data.response || '';
  } else if (provider === 'openai') {
    const customKey = llmConfig?.openaiApiKey;
    const endpoint = llmConfig?.openaiEndpoint || 'https://api.openai.com/v1';
    const model = llmConfig?.openaiModel || 'gpt-4o-mini';
    
    if (!customKey) {
      throw new Error('OpenAI API key is missing. Please configure it in the settings panel.');
    }
    
    const cleanEndpoint = endpoint.replace(/\/$/, '');
    const url = `${cleanEndpoint}/chat/completions`;
    
    const messages = [];
    if (systemInstruction) {
      messages.push({ role: 'system', content: systemInstruction });
    }
    messages.push({ role: 'user', content: prompt });
    
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${customKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.7,
      })
    });
    
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenAI request failed: ${res.statusText}. ${errText}`);
    }
    
    const data: any = await res.json();
    return data.choices?.[0]?.message?.content || '';
  } else {
    throw new Error(`Unknown LLM provider: ${provider}`);
  }
}

function formatLLMError(error: any, defaultText: string): string {
  const errMsg = error.message || '';
  if (errMsg.includes('503') || errMsg.includes('UNAVAILABLE') || errMsg.includes('high demand') || errMsg.includes('temporarily')) {
    return "### ⚠️ Gemini API High Demand\nThe Gemini model is currently experiencing extremely high demand. This is a temporary service-level limit. Please wait a few moments and try again!\n\n*Note: Your browser-based local Knowledge Graph Embedding (KGE) models, projection algorithms, and algebraic vectors are 100% active, fully functional, and unaffected.*";
  }
  if (errMsg.includes('429') || errMsg.includes('Quota')) {
    return "### ⚠️ Rate Limit Exceeded\nThe API key has temporarily exceeded its rate limits. Please try again in a minute.";
  }
  return `### Error generating explanation\n${errMsg || defaultText}`;
}

app.post('/api/gemini/interpret-relationship', async (req, res) => {
  try {
    const { sub, rel, obj, score, confidence, modelType, datasetDescription, subjectNeighborhood, objectNeighborhood, llmConfig } = req.body;
    
    if (!sub || !rel || !obj) {
      return res.status(400).json({ error: 'Missing triple elements (subject, relation, object).' });
    }

    let neighborhoodContext = '';
    if (subjectNeighborhood && subjectNeighborhood.length > 0) {
      neighborhoodContext += `\nExisting ground-truth connections involving Subject "${sub}" in the original graph:\n` +
        subjectNeighborhood.map((t: any) => `- (${t.sub}, ${t.rel}, ${t.obj})`).join('\n') + '\n';
    }
    if (objectNeighborhood && objectNeighborhood.length > 0) {
      neighborhoodContext += `\nExisting ground-truth connections involving Object "${obj}" in the original graph:\n` +
        objectNeighborhood.map((t: any) => `- (${t.sub}, ${t.rel}, ${t.obj})`).join('\n') + '\n';
    }

    let modelMathContext = '';
    const mt = (modelType || 'TransE').toLowerCase();
    if (mt.includes('transe')) {
      modelMathContext = `Model formulation is TransE (Translational Embeddings): f_r(h,t) = -||h + r - t||_2^2.
      Scores range from negative infinity to 0. A score closer to 0 (lower distance) represents a higher-probability translational match (perfect alignment is 0).`;
    } else if (mt.includes('distmult')) {
      modelMathContext = `Model formulation is DistMult (Bilinear Diagonal): f_r(h,t) = <h, r, t>.
      Scores are real values where higher positive numbers indicate stronger diagonal bilinear activation and semantic correlation.`;
    } else if (mt.includes('complex')) {
      modelMathContext = `Model formulation is ComplEx (Complex Bilinear): f_r(h,t) = Re(<h, r, t_bar>).
      Scores are real values where higher positive numbers represent higher probability of the asymmetric relationship, captured via the Hermitian inner product in the complex plane.`;
    } else if (mt.includes('rotate')) {
      modelMathContext = `Model formulation is RotatE (Rotational Embeddings): f_r(h,t) = -||h o r - t||_2^2.
      Entities and relations are represented in complex space, where relations rotate the head entity to align with the tail entity. Scores range from negative infinity to 0, where a score closer to 0 indicates excellent rotational alignment.`;
    }

    const systemInstruction = "You are an expert in Knowledge Graph Embeddings (KGE) and relational data science.";
    const prompt = `
      A KGE model (${modelType}) has analyzed a knowledge graph and predicted a *hidden relationship* that was NOT present in the original dataset.
      The model predicted this triple with a raw similarity score of ${score.toFixed(4)} and a normalized confidence of ${(confidence * 100).toFixed(1)}%.
      
      Mathematical scoring context for this specific embedding model:
      ${modelMathContext}
      
      Here is the predicted relationship:
      Subject: "${sub}"
      Relation: "${rel}"
      Object: "${obj}"
      
      The dataset represents: ${datasetDescription || 'a custom domain knowledge graph'}.
      ${neighborhoodContext}
      
      Tasks:
      1. Explain *why* the embedding model likely inferred this hidden connection based on standard semantic patterns (e.g., transitivity, symmetry, composition, or homophily). Frame your explanation with direct reference to any matching links or paths found in the provided local graph neighborhoods, and tie this explanation back to the KGE formulation described above (e.g., translational alignment, complex rotational rotation, or bilinear product).
      2. Interpret what this relationship means in the real world given the domain and the surrounding local context. Is this a logical discovery (e.g., path completion, role symmetry, or multi-hop composition) or a potential anomaly? Frame this interpretation carefully, avoiding ungrounded speculative assumptions about the domain, and explicitly point to the specific existing neighbors in the graph that support your reasoning.
      3. Provide a brief (2-3 sentences) hypothesis that could be tested in the real world to confirm this relationship.
      
      Provide a highly polished, informative, and professional response in markdown format. Keep it concise (under 250 words). Do not use dry jargon; speak clearly and objectively.
    `;

    const text = await generateLLMResponse(prompt, systemInstruction, llmConfig);
    res.json({ analysis: text });
  } catch (error: any) {
    console.error('LLM Relationship Interpretation Error:', error);
    res.status(500).json({ error: formatLLMError(error, 'Failed to generate relationship interpretation.') });
  }
});

// 2. Explain Latent Entity
app.post('/api/gemini/explain-latent-entity', async (req, res) => {
  try {
    const { name, sourceEntities, arithmeticOps, predictedRelations, datasetDescription, llmConfig } = req.body;
    
    if (!name || !sourceEntities || !predictedRelations) {
      return res.status(400).json({ error: 'Missing latent entity descriptors.' });
    }

    // Format predicted relations for the prompt
    let relsString = '';
    Object.keys(predictedRelations).forEach(rel => {
      const targets = predictedRelations[rel].map((t: any) => `"${t.target}" (score: ${t.score.toFixed(3)})`).join(', ');
      relsString += `- **${rel}**: ${targets}\n`;
    });

    const arithmeticString = sourceEntities.map((e: string, i: number) => {
      if (i === 0) return `"${e}"`;
      const op = arithmeticOps[i - 1] === 'sub' ? '-' : '+';
      return `${op} "${e}"`;
    }).join(' ');

    const systemInstruction = "You are a scientist exploring high-dimensional latent spaces.";
    const prompt = `
      A user has generated a *synthetic latent entity* named "${name}" in a Knowledge Graph Embedding model.
      This latent vector was constructed using the following embedding arithmetic:
      ${arithmeticString}
      
      The dataset represents: ${datasetDescription || 'a custom domain knowledge graph'}.
      
      Here are the top-scoring predicted relationships for this hypothetical entity in the embedding space:
      ${relsString}
      
      Tasks:
      1. Give a conceptual definition of what this hypothetical entity "${name}" represents in the real world. Why does the arithmetic combination of ${sourceEntities.join(', ')} result in these specific relations?
      2. Discuss the logical validity of this entity. Does it represent a realistic, hitherto unconfirmed entity (e.g., a new biological pathway, a combined market segment, a fictional hybrid character) or is it a conceptual paradox?
      3. Suggest how researchers or domain experts could "instantiate" or search for this entity in real-world data.
      
      Provide your response in beautifully formatted markdown under 300 words. Be objective, creative, and professional.
    `;

    const text = await generateLLMResponse(prompt, systemInstruction, llmConfig);
    res.json({ analysis: text });
  } catch (error: any) {
    console.error('LLM Latent Entity Explanation Error:', error);
    res.status(500).json({ error: formatLLMError(error, 'Failed to generate latent entity description.') });
  }
});

// 3. Global Model Analysis
app.post('/api/gemini/global-analysis', async (req, res) => {
  try {
    const { stats, modelType, datasetDescription, llmConfig } = req.body;
    
    if (!stats) {
      return res.status(400).json({ error: 'Missing model stats.' });
    }

    const systemInstruction = "You are an expert AI researcher writing a scientific summary.";
    const prompt = `
      A Knowledge Graph Embedding model of type "${modelType}" has successfully finished training.
      
      Here are the training and structural statistics:
      - Unique Entities: ${stats.numEntities}
      - Unique Relations: ${stats.numRelations}
      - Training Triples: ${stats.numTriples}
      - Final Training Loss: ${stats.finalLoss.toFixed(6)}
      - Projected Cluster Variance: ${stats.varianceExplained || 'Highly structured clusters'}
      
      The dataset represents: ${datasetDescription || 'a custom domain knowledge graph'}.
      
      Please write a high-level scientific summary (maximum 2 scannable paragraphs) describing:
      1. What the final loss of ${stats.finalLoss.toFixed(4)} indicates about the convergence and relational representation of the ${modelType} model on this specific graph.
      2. The architectural strengths of "${modelType}" (e.g., handling asymmetric relations, rotation phase, translational properties) and how those strengths mapped to this domain.
      3. A summary recommendation of how to best utilize this trained model for downstream tasks (like search, recommender engines, or anomaly detection).
      
      Provide a highly polished, executive response in clean markdown.
    `;

    const text = await generateLLMResponse(prompt, systemInstruction, llmConfig);
    res.json({ analysis: text });
  } catch (error: any) {
    console.error('LLM Global Analysis Error:', error);
    res.status(500).json({ error: formatLLMError(error, 'Failed to generate global embedding summary.') });
  }
});

// ==========================================
// Serve React App (Vite integration)
// ==========================================

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    // Development Mode: Mount Vite Dev Server
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Production Mode: Serve Static Bundle
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Knowledge Graph Embedding Explorer running on http://localhost:${PORT} [ENV: ${process.env.NODE_ENV || 'development'}]`);
  });
}

startServer();
