/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Main React Entry UI with responsive split layouts, interactive tabs, CSV uploading,
 * SVG-based real-time loss tracking, and full-stack Gemini API query handlers.
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import { 
  Network, Cpu, Layers, Play, Settings, RefreshCw, Sparkles, Plus, 
  HelpCircle, Activity, FileText, CheckCircle2, Search, Info, Trash2,
  ShieldAlert, AlertTriangle
} from 'lucide-react';
import { PRELOADED_DATASETS, Dataset } from './utils/datasets';
import { KGEEngine, Triple, KGEModelType, TrainingConfig, TrainingProgress, projectTo3D, projectTo3DWithUMAP, kmeansClustering } from './utils/kge';
import { Graph3D } from './components/Graph3D';

export default function App() {
  // Dataset State
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('medical-discovery');
  const [customTriples, setCustomTriples] = useState<Triple[] | null>(null);
  const [customName, setCustomName] = useState<string>('');
  const [customDesc, setCustomDesc] = useState<string>('');
  const [csvRaw, setCsvRaw] = useState<string>('');

  // CSV Configuration States
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [csvLines, setCsvLines] = useState<string[][]>([]);
  const [subColIndex, setSubColIndex] = useState<number>(0);
  const [relColIndex, setRelColIndex] = useState<number>(1);
  const [objColIndex, setObjColIndex] = useState<number>(2);
  const [csvHasHeaderRow, setCsvHasHeaderRow] = useState<boolean>(false);

  // active dataset compute
  const activeDataset = useMemo<Dataset>(() => {
    if (customTriples) {
      return {
        id: 'custom',
        name: customName || 'Uploaded Custom Dataset',
        description: customDesc || 'A user-provided knowledge graph uploaded via CSV.',
        triples: customTriples
      };
    }
    return PRELOADED_DATASETS.find(d => d.id === selectedDatasetId) || PRELOADED_DATASETS[0];
  }, [selectedDatasetId, customTriples, customName, customDesc]);

  // Model Parameters State
  const [modelType, setModelType] = useState<KGEModelType>('transE');
  const [dim, setDim] = useState<number>(50);
  const [lr, setLr] = useState<number>(0.02);
  const [margin, setMargin] = useState<number>(4.0);
  const [epochs, setEpochs] = useState<number>(200);
  const [negSamples, setNegSamples] = useState<number>(2);

  // Training & KGE Engine States
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentProgress, setCurrentProgress] = useState<TrainingProgress | null>(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [engineInstance, setEngineInstance] = useState<KGEEngine | null>(null);
  const [statusText, setStatusText] = useState<string>('Ready for training');

  // UI Selection Coordinates
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'link' | 'relations' | 'latent' | 'metrics'>('link');

  // Projection & Clustering Options
  const [projectionMode, setProjectionMode] = useState<'pca' | 'umap'>('pca');
  const [clusteringMode, setClusteringMode] = useState<'default' | 'kmeans'>('kmeans');
  const [numClusters, setNumClusters] = useState<number>(5);

  // Link Prediction Form State
  const [predSub, setPredSub] = useState<string>('');
  const [predRel, setPredRel] = useState<string>('');
  const [predObj, setPredObj] = useState<string>('');
  const [predResults, setPredResults] = useState<{ target: string; score: number; probability: number }[]>([]);

  // Latent Entity Generator State
  const [latentName, setLatentName] = useState<string>('Synthetic_Concept_Alpha');
  const [latentSources, setLatentSources] = useState<string[]>(['', '', '']);
  const [latentOps, setLatentOps] = useState<('add' | 'sub')[]>(['add', 'sub']);
  const [latentRelations, setLatentRelations] = useState<{ [rel: string]: { target: string; score: number }[] }>({});
  const [latentSortOrder, setLatentSortOrder] = useState<'desc' | 'asc' | 'alphabetical'>('desc');
  const [latentAnalysis, setLatentAnalysis] = useState<string>('');
  const [isGeneratingLatentConcept, setIsGeneratingLatentConcept] = useState<boolean>(false);

  // Gemini Explanation Outputs
  const [selectedTripleExplanation, setSelectedTripleExplanation] = useState<{ sub: string; rel: string; obj: string; text: string } | null>(null);
  const [isGeneratingExplanation, setIsGeneratingExplanation] = useState<boolean>(false);
  const [globalAnalysisText, setGlobalAnalysisText] = useState<string>('');
  const [isGeneratingGlobalAnalysis, setIsGeneratingGlobalAnalysis] = useState<boolean>(false);

  // LLM Configurations & Settings Cogwheel
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [llmConfig, setLlmConfig] = useState<{
    provider: 'gemini' | 'ollama' | 'openai';
    geminiApiKey: string;
    ollamaEndpoint: string;
    ollamaModel: string;
    openaiApiKey: string;
    openaiEndpoint: string;
    openaiModel: string;
  }>(() => {
    const saved = localStorage.getItem('kge_llm_config');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return {
          provider: parsed.provider || 'gemini',
          geminiApiKey: parsed.geminiApiKey || '',
          ollamaEndpoint: parsed.ollamaEndpoint || 'http://localhost:11434',
          ollamaModel: parsed.ollamaModel || 'gemma2',
          openaiApiKey: parsed.openaiApiKey || '',
          openaiEndpoint: parsed.openaiEndpoint || 'https://api.openai.com/v1',
          openaiModel: parsed.openaiModel || 'gpt-4o-mini',
        };
      } catch (e) {}
    }
    return {
      provider: 'gemini',
      geminiApiKey: '',
      ollamaEndpoint: 'http://localhost:11434',
      ollamaModel: 'gemma2',
      openaiApiKey: '',
      openaiEndpoint: 'https://api.openai.com/v1',
      openaiModel: 'gpt-4o-mini',
    };
  });

  // Save config on changes
  useEffect(() => {
    localStorage.setItem('kge_llm_config', JSON.stringify(llmConfig));
  }, [llmConfig]);

  // Data Hygiene / OOD State
  const [hygieneFilter, setHygieneFilter] = useState<'anomalies' | 'all'>('anomalies');
  const [hygieneSearch, setHygieneSearch] = useState<string>('');

  // Trained configuration tracker
  const [trainedConfig, setTrainedConfig] = useState<{ modelType: KGEModelType; dim: number } | null>(null);

  // Synthetic Concepts tracker
  const [syntheticConcepts, setSyntheticConcepts] = useState<{ name: string; sourceEntities: string[]; ops: ('add' | 'sub')[] }[]>([]);

  // Derived indicator: is the loaded model trained for the currently selected config?
  const isModelTrainedForSelectedConfig = useMemo(() => {
    return engineInstance !== null && 
           trainedConfig !== null && 
           trainedConfig.modelType === modelType && 
           trainedConfig.dim === dim;
  }, [engineInstance, trainedConfig, modelType, dim]);

  // Combined visualizer entity embeddings including synthetic concepts
  const visualizerEntityEmbeddings = useMemo(() => {
    if (!engineInstance || !currentProgress || !isModelTrainedForSelectedConfig) return null;
    
    // Get raw trained embeddings
    const fullEnt = engineInstance.getEntityEmbeddingsDict(modelType, dim);
    
    // Add synthetic concepts
    syntheticConcepts.forEach(sc => {
      const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
      const newVec = new Float32Array(size);
      
      if (sc.sourceEntities.length > 0) {
        const baseIdx = engineInstance.entity2idx[sc.sourceEntities[0]];
        if (baseIdx !== undefined) {
          const baseVec = engineInstance.getEntityVector(baseIdx, size);
          for (let i = 0; i < size; i++) newVec[i] = baseVec[i];
          
          for (let i = 1; i < sc.sourceEntities.length; i++) {
            const idx = engineInstance.entity2idx[sc.sourceEntities[i]];
            if (idx !== undefined) {
              const vec = engineInstance.getEntityVector(idx, size);
              const op = sc.ops[i - 1] || 'add';
              for (let d = 0; d < size; d++) {
                if (op === 'add') {
                  newVec[d] += vec[d];
                } else {
                  newVec[d] -= vec[d];
                }
              }
            }
          }
          
          fullEnt[sc.name] = Array.from(newVec);
        }
      }
    });
    
    return projectionMode === 'umap' ? projectTo3DWithUMAP(fullEnt) : projectTo3D(fullEnt);
  }, [engineInstance, currentProgress, isModelTrainedForSelectedConfig, modelType, dim, syntheticConcepts, projectionMode]);

  // Compute high-dimensional K-Means clusters for structural coloring
  const kmeansClusterAssignments = useMemo(() => {
    if (!engineInstance || !isModelTrainedForSelectedConfig || clusteringMode !== 'kmeans') return null;
    
    // Get raw trained embeddings
    const fullEnt = engineInstance.getEntityEmbeddingsDict(modelType, dim);
    
    // Add synthetic concepts
    syntheticConcepts.forEach(sc => {
      const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
      const newVec = new Float32Array(size);
      
      if (sc.sourceEntities.length > 0) {
        const baseIdx = engineInstance.entity2idx[sc.sourceEntities[0]];
        if (baseIdx !== undefined) {
          const baseVec = engineInstance.getEntityVector(baseIdx, size);
          for (let i = 0; i < size; i++) newVec[i] = baseVec[i];
          
          for (let i = 1; i < sc.sourceEntities.length; i++) {
            const idx = engineInstance.entity2idx[sc.sourceEntities[i]];
            if (idx !== undefined) {
              const vec = engineInstance.getEntityVector(idx, size);
              const op = sc.ops[i - 1] || 'add';
              for (let d = 0; d < size; d++) {
                if (op === 'add') {
                  newVec[d] += vec[d];
                } else {
                  newVec[d] -= vec[d];
                }
              }
            }
          }
          
          fullEnt[sc.name] = Array.from(newVec);
        }
      }
    });

    return kmeansClustering(fullEnt, numClusters);
  }, [engineInstance, isModelTrainedForSelectedConfig, clusteringMode, modelType, dim, numClusters, syntheticConcepts]);

  // Generate nodeColors dictionary mapping for Graph3D
  const nodeColors = useMemo(() => {
    if (!kmeansClusterAssignments || clusteringMode !== 'kmeans') return undefined;
    
    const clusterGradients = [
      { start: '#3b82f6', end: '#1d4ed8' }, // Blue
      { start: '#10b981', end: '#047857' }, // Emerald/Teal
      { start: '#f97316', end: '#c2410c' }, // Orange
      { start: '#ec4899', end: '#be185d' }, // Pink/Rose
      { start: '#eab308', end: '#a16207' }, // Yellow/Gold
      { start: '#8b5cf6', end: '#6d28d9' }, // Violet
      { start: '#06b6d4', end: '#0891b2' }, // Cyan
      { start: '#ef4444', end: '#b91c1c' }, // Red
      { start: '#84cc16', end: '#4d7c0f' }, // Lime Green
      { start: '#64748b', end: '#334155' }, // Slate
    ];

    const colorsMap: { [entityName: string]: { start: string; end: string } } = {};
    Object.keys(kmeansClusterAssignments).forEach(entName => {
      const clusterId = kmeansClusterAssignments[entName];
      const grad = clusterGradients[clusterId % clusterGradients.length];
      colorsMap[entName] = grad;
    });

    return colorsMap;
  }, [kmeansClusterAssignments, clusteringMode]);

  // Combined visualizer triples including virtual relations for synthetic concepts
  const visualizerTriples = useMemo(() => {
    const baseTriples = [...activeDataset.triples];
    
    if (!engineInstance || !isModelTrainedForSelectedConfig) return baseTriples;
    
    // For each synthetic concept, let's add its top predictions as virtual triples!
    syntheticConcepts.forEach(sc => {
      // Calculate predictions
      const preds = engineInstance.generateLatentEntity(
        sc.name,
        sc.sourceEntities,
        sc.ops,
        modelType,
        dim
      );
      
      // For each relation, find the top predicted target and add a virtual triple
      Object.keys(preds).forEach(rel => {
        const topObj = preds[rel][0];
        if (topObj && topObj.score > -15.0) { // Only add relatively strong connections
          baseTriples.push({
            sub: sc.name,
            rel: rel,
            obj: topObj.target
          });
        }
      });
    });
    
    return baseTriples;
  }, [activeDataset, engineInstance, isModelTrainedForSelectedConfig, syntheticConcepts, modelType, dim]);

  // Inline unique entity lists
  const uniqueEntities = useMemo(() => {
    const set = new Set<string>();
    activeDataset.triples.forEach(t => {
      set.add(t.sub);
      set.add(t.obj);
    });
    return Array.from(set).sort();
  }, [activeDataset]);

  const uniqueRelations = useMemo(() => {
    const set = new Set<string>();
    activeDataset.triples.forEach(t => set.add(t.rel));
    return Array.from(set).sort();
  }, [activeDataset]);

  // Set default prediction fields on dataset swap
  useEffect(() => {
    if (uniqueEntities.length > 0) setPredSub(uniqueEntities[0]);
    if (uniqueRelations.length > 0) setPredRel(uniqueRelations[0]);
    setPredObj('');
    setPredResults([]);
    setSelectedNode(null);
    setEngineInstance(null);
    setCurrentProgress(null);
    setLossHistory([]);
    setTrainedConfig(null);
    setSyntheticConcepts([]);
    setLatentSources([uniqueEntities[0] || '', uniqueEntities[1] || '', uniqueEntities[2] || '']);
    setLatentOps(['add', 'sub']);
    setLatentRelations({});
    setLatentAnalysis('');
    setGlobalAnalysisText('');
  }, [activeDataset, uniqueEntities, uniqueRelations]);

  // Handle local nearest neighbors
  const nearestNeighbors = useMemo(() => {
    if (!isModelTrainedForSelectedConfig || isTraining || !selectedNode || !engineInstance) return [];
    return engineInstance.getNearestNeighbors(selectedNode, modelType, dim);
  }, [isModelTrainedForSelectedConfig, isTraining, selectedNode, modelType, dim, engineInstance]);

  // Real-time Discovered Latent Triples
  const discoveredLatentTriples = useMemo(() => {
    if (!isModelTrainedForSelectedConfig || isTraining || !engineInstance) return [];
    return engineInstance.discoverLatentRelations(modelType, dim, 10);
  }, [isModelTrainedForSelectedConfig, isTraining, modelType, dim, engineInstance]);

  // Real-time Link Prediction Quality Metrics (MRR, Hits@1, Hits@5, Hits@10)
  const kgeMetrics = useMemo(() => {
    if (!isModelTrainedForSelectedConfig || isTraining || !engineInstance) return null;
    return engineInstance.evaluateMetrics(modelType, dim);
  }, [isModelTrainedForSelectedConfig, isTraining, modelType, dim, engineInstance]);

  // Real-time Data Hygiene / Anomaly Detection (OOD)
  const oodAnomalies = useMemo(() => {
    if (!isModelTrainedForSelectedConfig || isTraining || !engineInstance) return [];
    return engineInstance.detectAnomalies(modelType, dim);
  }, [isModelTrainedForSelectedConfig, isTraining, modelType, dim, engineInstance]);

  // Real-time CSV Parsing & Auto-detection Effect
  useEffect(() => {
    if (!csvRaw.trim()) {
      setCsvHeaders([]);
      setCsvLines([]);
      setCsvHasHeaderRow(false);
      return;
    }

    try {
      const parseCSVLine = (line: string): string[] => {
        const result: string[] = [];
        let current = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
          const char = line[i];
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            result.push(current.trim());
            current = '';
          } else {
            current += char;
          }
        }
        result.push(current.trim());
        return result.map(field => {
          let f = field.trim();
          if (f.startsWith('"') && f.endsWith('"')) {
            f = f.substring(1, f.length - 1);
          } else if (f.startsWith("'") && f.endsWith("'")) {
            f = f.substring(1, f.length - 1);
          }
          return f;
        });
      };

      const lines = csvRaw.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      if (lines.length === 0) return;

      const parsedLines = lines.map(line => parseCSVLine(line));
      setCsvLines(parsedLines);

      // Detect header row by checking keywords in the first line
      const firstLine = lines[0].toLowerCase();
      const isHeader = firstLine.includes('sub') || firstLine.includes('rel') || firstLine.includes('obj') || 
                       firstLine.includes('head') || firstLine.includes('tail') || firstLine.includes('source') || 
                       firstLine.includes('target') || firstLine.includes('from') || firstLine.includes('to');
      
      setCsvHasHeaderRow(isHeader);

      let detectedHeaders: string[] = [];
      if (isHeader) {
        detectedHeaders = parsedLines[0];
      } else {
        // Generate placeholder headers
        const maxCols = Math.max(...parsedLines.map(l => l.length));
        detectedHeaders = Array.from({ length: maxCols }, (_, i) => `Col ${i + 1}`);
      }
      setCsvHeaders(detectedHeaders);

      // Auto-detect mappings based on headers
      const lowerHeaders = detectedHeaders.map(h => h.toLowerCase());
      let subIdx = lowerHeaders.findIndex(h => h.includes('sub') || h.includes('head') || h.includes('from') || h.includes('source') || h === 'h' || h === 's');
      let relIdx = lowerHeaders.findIndex(h => h.includes('rel') || h.includes('type') || h.includes('edge') || h.includes('relation') || h === 'r' || h === 'p');
      let objIdx = lowerHeaders.findIndex(h => h.includes('obj') || h.includes('tail') || h.includes('to') || h.includes('target') || h === 't' || h === 'o');

      if (subIdx === -1) subIdx = 0;
      if (relIdx === -1) relIdx = 1;
      if (objIdx === -1) objIdx = 2;

      // Ensure indices are within boundary
      const numCols = detectedHeaders.length;
      if (subIdx >= numCols) subIdx = 0;
      if (relIdx >= numCols) relIdx = Math.min(1, numCols - 1);
      if (objIdx >= numCols) objIdx = Math.min(2, numCols - 1);

      setSubColIndex(subIdx);
      setRelColIndex(relIdx);
      setObjColIndex(objIdx);
    } catch (e) {
      // Allow user to keep typing
    }
  }, [csvRaw]);

  // Dynamic CSV Upload Parsing
  const handleCSVUpload = (text: string) => {
    if (csvLines.length === 0) return;
    
    try {
      const parsed: Triple[] = [];
      const startIndex = csvHasHeaderRow ? 1 : 0;

      for (let i = startIndex; i < csvLines.length; i++) {
        const cols = csvLines[i];
        if (cols.length > Math.max(subColIndex, relColIndex, objColIndex)) {
          const sub = cols[subColIndex] || '';
          const rel = cols[relColIndex] || '';
          const obj = cols[objColIndex] || '';
          if (sub && rel && obj) {
            parsed.push({
              id: `custom_${i}`,
              sub,
              rel,
              obj
            });
          }
        }
      }

      if (parsed.length === 0) {
        alert('Could not find any valid triples. Adjust column mapping to match your CSV columns.');
        return;
      }

      setCustomTriples(parsed);
      setCustomName('Custom CSV Knowledge Graph');
      setCustomDesc(`Uploaded via file parser. Contains ${parsed.length} triples.`);
      setStatusText('Custom CSV loaded. Ready to train!');
    } catch (err: any) {
      alert('Error parsing CSV file: ' + err.message);
    }
  };

  const clearCustomDataset = () => {
    setCustomTriples(null);
    setCustomName('');
    setCustomDesc('');
    setCsvRaw('');
    setSelectedDatasetId('medical-discovery');
  };

  // Launch training engine
  const handleTrain = () => {
    if (activeDataset.triples.length === 0) return;
    
    setIsTraining(true);
    setTrainedConfig(null); // Clear trained state during active training
    setLossHistory([]);
    setGlobalAnalysisText('');
    setStatusText('Initializing parameters...');

    const config: TrainingConfig = {
      modelType,
      dim,
      lr,
      margin,
      epochs,
      negSamples,
    };

    const tempLosses: number[] = [];

    // Instantiate and launch training thread in steps to avoid blocking main thread
    const engine = new KGEEngine(activeDataset.triples);
    engine.initialize(config.modelType, config.dim);
    setEngineInstance(engine);

    let currentEpoch = 0;

    const step = () => {
      if (currentEpoch >= config.epochs) {
        setIsTraining(false);
        setStatusText('Training completed successfully!');
        setTrainedConfig({ modelType: config.modelType, dim: config.dim });
        return;
      }

      const loss = engine.trainEpoch(config);
      tempLosses.push(loss);
      setLossHistory([...tempLosses]);
      currentEpoch++;

      // Compute 3D Coordinates via Power Iteration (Deflation SVD/PCA)
      // Done every 10 epochs or on final epoch to preserve canvas rendering performance
      if (currentEpoch % 10 === 0 || currentEpoch === config.epochs) {
        const fullEnt = engine.getEntityEmbeddingsDict(config.modelType, config.dim);
        const fullRel = engine.getRelationEmbeddingsDict(config.modelType, config.dim);
        
        // Import our fast projectTo3D math
        const projEnt = projectTo3D(fullEnt);
        const projRel = projectTo3D(fullRel);

        setCurrentProgress({
          epoch: currentEpoch,
          loss,
          entityEmbeddings: projEnt,
          relationEmbeddings: projRel,
        });
      }

      setStatusText(`Training: Epoch ${currentEpoch}/${config.epochs} • Current Loss: ${loss.toFixed(5)}`);

      if (currentEpoch < config.epochs) {
        // Queue next frame instantly
        requestAnimationFrame(step);
      } else {
        setIsTraining(false);
        setStatusText('Training completed successfully!');
        setTrainedConfig({ modelType: config.modelType, dim: config.dim });
      }
    };

    // Begin asynchronous animation loop
    requestAnimationFrame(step);
  };

  // Handle Link Prediction Scoring
  const handlePredict = () => {
    if (!engineInstance) {
      alert('Please train the embedding model first!');
      return;
    }

    // Determine query format
    const s = predSub || null;
    const r = predRel || null;
    const o = predObj || null;

    const results = engineInstance.predictLinks(s, r, o, modelType, dim, 12);
    setPredResults(results);
  };

  // Explain Discovered Relation using Full-Stack Gemini Route
  const handleExplainRelation = async (sub: string, rel: string, obj: string, score: number, confidence: number) => {
    setIsGeneratingExplanation(true);
    setSelectedTripleExplanation({ sub, rel, obj, text: `Consulting ${llmConfig.provider === 'gemini' ? 'Gemini' : llmConfig.provider === 'ollama' ? 'Ollama' : 'OpenAI'} model for latent space analysis...` });
    
    try {
      const response = await fetch('/api/gemini/interpret-relationship', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sub,
          rel,
          obj,
          score,
          confidence,
          modelType,
          datasetDescription: activeDataset.description,
          llmConfig
        })
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setSelectedTripleExplanation({ sub, rel, obj, text: data.analysis });
    } catch (err: any) {
      const errMsg = err.message || '';
      setSelectedTripleExplanation({
        sub,
        rel,
        obj,
        text: errMsg.startsWith('###') ? errMsg : `### Error generating explanation\n${errMsg || 'The server-side Gemini request failed.'}`
      });
    } finally {
      setIsGeneratingExplanation(false);
    }
  };

  // Generate Latent Entity Predicted Relations
  const handleCalculateLatentEntity = () => {
    if (!engineInstance) {
      alert('Please train the embedding model first!');
      return;
    }

    const filteredSources = latentSources.filter(s => s !== '');
    if (filteredSources.length === 0) {
      alert('Please select at least one base entity.');
      return;
    }

    const predictions = engineInstance.generateLatentEntity(
      latentName,
      filteredSources,
      latentOps,
      modelType,
      dim
    );

    setLatentRelations(predictions);
    setLatentAnalysis('');

    // Save synthetic concept to state so it gets projected and rendered in the graph
    setSyntheticConcepts(prev => {
      const filtered = prev.filter(c => c.name !== latentName);
      return [...filtered, {
        name: latentName,
        sourceEntities: filteredSources,
        ops: [...latentOps],
      }];
    });
  };

  // Delete a synthetic concept
  const handleDeleteSyntheticConcept = (name: string) => {
    setSyntheticConcepts(prev => prev.filter(c => c.name !== name));
    if (latentName === name) {
      setLatentRelations({});
      setLatentAnalysis('');
    }
  };

  // Explain Latent Entity via Server-Side Gemini API
  const handleExplainLatentEntity = async () => {
    if (Object.keys(latentRelations).length === 0) {
      alert('Please calculate the latent relations first.');
      return;
    }

    setIsGeneratingLatentConcept(true);
    setLatentAnalysis(`Synthesizing conceptual meaning of latent space coords with ${llmConfig.provider === 'gemini' ? 'Gemini' : llmConfig.provider === 'ollama' ? 'Ollama' : 'OpenAI'}...`);

    try {
      const response = await fetch('/api/gemini/explain-latent-entity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: latentName,
          sourceEntities: latentSources.filter(s => s !== ''),
          arithmeticOps: latentOps,
          predictedRelations: latentRelations,
          datasetDescription: activeDataset.description,
          llmConfig
        })
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setLatentAnalysis(data.analysis);
    } catch (err: any) {
      const errMsg = err.message || '';
      setLatentAnalysis(errMsg.startsWith('###') ? errMsg : `### Conceptual Synthesis Failed\n${errMsg}`);
    } finally {
      setIsGeneratingLatentConcept(false);
    }
  };

  // Explain Global Topology via Gemini
  const handleGlobalAnalysis = async () => {
    if (!engineInstance) return;

    setIsGeneratingGlobalAnalysis(true);
    setGlobalAnalysisText(`Compiling global embedding metrics and querying ${llmConfig.provider === 'gemini' ? 'Gemini' : llmConfig.provider === 'ollama' ? 'Ollama' : 'OpenAI'}...`);

    try {
      const stats = {
        numEntities: uniqueEntities.length,
        numRelations: uniqueRelations.length,
        numTriples: activeDataset.triples.length,
        finalLoss: lossHistory[lossHistory.length - 1] || 0.0,
      };

      const response = await fetch('/api/gemini/global-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stats,
          modelType,
          datasetDescription: activeDataset.description,
          llmConfig
        })
      });

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setGlobalAnalysisText(data.analysis);
    } catch (err: any) {
      const errMsg = err.message || '';
      setGlobalAnalysisText(errMsg.startsWith('###') ? errMsg : `### Global analysis compilation failed\n${errMsg}`);
    } finally {
      setIsGeneratingGlobalAnalysis(false);
    }
  };

  // Math vector visual representation coordinates for real-time model values
  const getEmbeddingsValueStr = (ent: string) => {
    if (!engineInstance) return '';
    const idx = engineInstance.entity2idx[ent];
    if (idx === undefined) return '';
    const size = (modelType === 'transE' || modelType === 'distMult') ? dim : dim * 2;
    const vec = engineInstance.getEntityVector(idx, size);
    return `[${Array.from(vec.subarray(0, 4)).map((v: any) => (v as number).toFixed(3)).join(', ')}...]`;
  };

  // SVG Line Chart Compute for real-time learning curve (sparkline)
  const renderSVGLossChart = () => {
    if (lossHistory.length === 0) return null;

    const width = 340;
    const height = 120;
    const padding = 15;

    const minLoss = Math.min(...lossHistory);
    const maxLoss = Math.max(...lossHistory);
    const lossRange = maxLoss - minLoss || 1.0;

    const points = lossHistory.map((loss, idx) => {
      const x = padding + (idx / (lossHistory.length - 1 || 1)) * (width - 2 * padding);
      const y = height - padding - ((loss - minLoss) / lossRange) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg className="w-full h-[120px] rounded-xl bg-[#020408] border border-white/5 mt-2 p-1" viewBox={`0 0 ${width} ${height}`}>
        {/* Grid lines */}
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" strokeDasharray="3" />
        <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="rgba(255, 255, 255, 0.05)" strokeWidth="1" strokeDasharray="3" />
        
        {/* Trend Area Gradient */}
        <defs>
          <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {lossHistory.length > 1 && (
          <path
            d={`M ${padding},${height - padding} L ${points} L ${width - padding},${height - padding} Z`}
            fill="url(#chartGrad)"
          />
        )}

        {/* The Line */}
        <polyline
          fill="none"
          stroke="#06b6d4"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />

        {/* Labels */}
        <text x={padding} y={height - 4} className="text-[9px] font-mono fill-slate-500">EPOCH 0</text>
        <text x={width - padding} y={height - 4} textAnchor="end" className="text-[9px] font-mono fill-slate-500">EP {lossHistory.length}</text>
        <text x={padding - 2} y={padding + 6} className="text-[9px] font-mono fill-cyan-400">MAX: {maxLoss.toFixed(3)}</text>
        <text x={padding - 2} y={height - padding - 2} className="text-[9px] font-mono fill-slate-500">MIN: {minLoss.toFixed(3)}</text>
      </svg>
    );
  };

  return (
    <div id="app-root-frame" className="min-h-screen bg-[#f8fafc] text-slate-700 flex flex-col font-sans selection:bg-indigo-100 selection:text-indigo-900">
      {/* Dynamic Header */}
      <header id="app-nav-header" className="w-full border-b border-slate-200 bg-white/90 backdrop-blur-md sticky top-0 z-40 px-6 py-4 shadow-xs">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-indigo-500 to-indigo-600 text-white shadow-md shadow-indigo-500/10">
              <Network className="h-6 w-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-[0.05em] text-slate-900 font-sans uppercase flex items-center gap-2">
                EIDOS <span className="text-slate-350 font-normal">//</span> KGE EXPLORER
              </h1>
              <p className="text-xs text-slate-500 font-mono tracking-wide">
                Project hidden relationships and explore latent vector arithmetic
              </p>
            </div>
          </div>
          
          <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-wider">
            <span 
              className="px-2.5 py-1 rounded-md bg-amber-50 border border-amber-200/85 text-amber-700 flex items-center gap-1 font-bold cursor-help shadow-xs"
              title="All core Knowledge Graph Embedding models, 3D/2D layouts, and latent vector arithmetic run 100% locally in your browser. Active LLM/Gemini use is strictly optional for semantic explanation features!"
            >
              <AlertTriangle className="h-3 w-3 text-amber-600 animate-pulse" /> ! LLM OPTIONAL
            </span>
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="px-2.5 py-1 rounded-md bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 hover:text-slate-800 flex items-center gap-1 cursor-pointer transition-colors shadow-xs"
              title="Configure LLM Providers"
            >
              <Settings className="h-3 w-3 text-slate-500" />
              <span>SETTINGS</span>
            </button>
            <span className="px-2.5 py-1 rounded-md bg-white border border-slate-200/80 shadow-xs flex items-center gap-1.5 text-slate-600">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-ping"></span>
              DATA: <b className="text-slate-800">{activeDataset.id}</b>
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white border border-slate-200/80 shadow-xs flex items-center gap-1.5 text-slate-600">
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-400"></span>
              MODEL: <b className="text-slate-800">{modelType}</b>
            </span>
            <span className="px-2.5 py-1 rounded-md bg-white border border-slate-200/80 shadow-xs flex items-center gap-1.5 text-slate-600">
              STATUS: <b className="text-indigo-600">{statusText}</b>
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main id="app-main-content" className="flex-grow w-full max-w-7xl mx-auto p-6 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full">
          
          {/* Left Panel: Configuration & Parameters (col-span-4) */}
          <div className="lg:col-span-4 flex flex-col gap-6">
            
            {/* Dataset Selection & Upload */}
            <div id="dataset-picker-card" className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-md shadow-slate-100 flex flex-col gap-4">
              <h2 className="text-[10px] font-bold tracking-[0.2em] text-slate-500 uppercase flex items-center gap-2 font-mono mb-1">
                <Layers className="h-3.5 w-3.5 text-indigo-500" /> DATASET MANAGER
              </h2>
              
              <div className="flex flex-col gap-2">
                <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono">Select Relational Graph</label>
                <select
                  value={customTriples ? 'custom' : selectedDatasetId}
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val === 'custom') {
                      // Trigger custom modal/form state
                    } else {
                      setCustomTriples(null);
                      setSelectedDatasetId(val);
                    }
                  }}
                  className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 focus:outline-none focus:border-indigo-500/40 font-mono"
                >
                  {PRELOADED_DATASETS.map(d => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                  {customTriples && <option value="custom">★ {customName}</option>}
                </select>
              </div>

              {/* CSV Manual Upload Widget */}
              <div className="p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/60 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider font-bold text-slate-600 flex items-center gap-1 font-mono">
                    Upload Custom KG (.csv)
                  </span>
                  {customTriples && (
                    <button
                      onClick={clearCustomDataset}
                      className="px-2 py-0.5 text-[9px] bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-600 font-mono rounded transition-all cursor-pointer flex items-center gap-1"
                      title="Clear Custom Dataset"
                    >
                      <Trash2 className="h-3 w-3" /> CLEAR
                    </button>
                  )}
                </div>
                
                <textarea
                  placeholder="Subject,Relation,Object&#10;Alice,sisterOf,Bob&#10;Bob,livesIn,London"
                  value={csvRaw}
                  onChange={(e) => setCsvRaw(e.target.value)}
                  className="w-full h-[70px] bg-white border border-slate-200 hover:border-slate-300 rounded-lg p-2 text-[10px] font-mono text-slate-700 focus:outline-none focus:border-indigo-500/40 resize-y"
                />

                {csvLines.length > 0 && (
                  <div className="p-2.5 bg-white border border-slate-200 rounded-lg flex flex-col gap-2 font-mono text-[10px] text-slate-700 shadow-inner">
                    <div className="font-bold text-indigo-600 uppercase text-[9px] tracking-wider border-b border-slate-100 pb-1 mb-1">
                      Configure Column Mapping
                    </div>

                    <div className="flex items-center gap-2 mb-1">
                      <input 
                        type="checkbox" 
                        id="hasHeaderCheckbox"
                        checked={csvHasHeaderRow}
                        onChange={(e) => setCsvHasHeaderRow(e.target.checked)}
                        className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 h-3.5 w-3.5"
                      />
                      <label htmlFor="hasHeaderCheckbox" className="text-[9px] font-bold text-slate-500 cursor-pointer uppercase">
                        First row is header
                      </label>
                    </div>

                    <div className="grid grid-cols-3 gap-1.5">
                      <div className="flex flex-col gap-1">
                        <span className="text-slate-400 text-[9px] font-bold">SUBJECT</span>
                        <select 
                          value={subColIndex} 
                          onChange={(e) => setSubColIndex(Number(e.target.value))}
                          className="p-1 bg-slate-50 border border-slate-200 rounded text-[10px] text-slate-800 focus:outline-none focus:border-indigo-500"
                        >
                          {csvHeaders.map((h, i) => (
                            <option key={i} value={i}>{`Col ${i}: ${h}`}</option>
                          ))}
                        </select>
                      </div>

                      <div className="flex flex-col gap-1">
                        <span className="text-slate-400 text-[9px] font-bold">RELATION</span>
                        <select 
                          value={relColIndex} 
                          onChange={(e) => setRelColIndex(Number(e.target.value))}
                          className="p-1 bg-slate-50 border border-slate-200 rounded text-[10px] text-slate-800 focus:outline-none focus:border-indigo-500"
                        >
                          {csvHeaders.map((h, i) => (
                            <option key={i} value={i}>{`Col ${i}: ${h}`}</option>
                          ))}
                        </select>
                      </div>

                      <div className="flex flex-col gap-1">
                        <span className="text-slate-400 text-[9px] font-bold">OBJECT</span>
                        <select 
                          value={objColIndex} 
                          onChange={(e) => setObjColIndex(Number(e.target.value))}
                          className="p-1 bg-slate-50 border border-slate-200 rounded text-[10px] text-slate-800 focus:outline-none focus:border-indigo-500"
                        >
                          {csvHeaders.map((h, i) => (
                            <option key={i} value={i}>{`Col ${i}: ${h}`}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Real-time sample preview row */}
                    {csvLines.length > (csvHasHeaderRow ? 1 : 0) && (
                      <div className="mt-1 p-2 bg-indigo-50/50 border border-indigo-100 rounded text-[9px] flex flex-col gap-1">
                        <div className="text-indigo-700/80 font-bold uppercase tracking-wider text-[8px]">First Triple Preview:</div>
                        <div className="flex flex-wrap items-center gap-1 font-bold text-slate-700">
                          <span className="bg-white px-1.5 py-0.5 rounded border border-slate-150 shadow-2xs max-w-[100px] truncate">
                            {csvLines[csvHasHeaderRow ? 1 : 0][subColIndex] || 'N/A'}
                          </span>
                          <span className="text-indigo-600 truncate max-w-[80px]">
                            -{csvLines[csvHasHeaderRow ? 1 : 0][relColIndex] || 'N/A'}→
                          </span>
                          <span className="bg-white px-1.5 py-0.5 rounded border border-slate-150 shadow-2xs max-w-[100px] truncate">
                            {csvLines[csvHasHeaderRow ? 1 : 0][objColIndex] || 'N/A'}
                          </span>
                        </div>
                      </div>
                    )}

                    <button
                      onClick={() => handleCSVUpload(csvRaw)}
                      className="w-full py-2 mt-1 bg-indigo-600 hover:bg-indigo-700 text-white font-bold font-mono text-[10px] tracking-wider uppercase rounded transition-all cursor-pointer shadow-sm text-center"
                    >
                      LOAD {csvLines.length - (csvHasHeaderRow ? 1 : 0)} TRIPLES
                    </button>
                  </div>
                )}
              </div>

              {/* Topology Summary */}
              <div className="grid grid-cols-3 gap-2 font-mono text-center text-xs">
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-200/60">
                  <span className="text-[9px] text-slate-500 block tracking-wider">TRIPLES</span>
                  <span className="text-sm font-bold text-slate-800">{activeDataset.triples.length}</span>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-200/60">
                  <span className="text-[9px] text-slate-500 block tracking-wider">ENTITIES</span>
                  <span className="text-sm font-bold text-indigo-600">{uniqueEntities.length}</span>
                </div>
                <div className="p-2 rounded-lg bg-slate-50 border border-slate-200/60">
                  <span className="text-[9px] text-slate-500 block tracking-wider">RELATIONS</span>
                  <span className="text-sm font-bold text-blue-600">{uniqueRelations.length}</span>
                </div>
              </div>
            </div>

            {/* Model Embedder Parameters */}
            <div id="model-parameters-card" className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-md shadow-slate-100 flex flex-col gap-4">
              <h2 className="text-[10px] font-bold tracking-[0.2em] text-slate-500 uppercase flex items-center gap-2 font-mono mb-1">
                <Cpu className="h-3.5 w-3.5 text-indigo-500" /> KGE MODEL PARAMETERS
              </h2>

              {/* Model Select */}
              <div className="flex flex-col gap-2">
                <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono">Embedding Formulation</label>
                <div className="grid grid-cols-2 gap-1.5 font-mono text-[11px]">
                  {(['transE', 'distMult', 'complEx', 'rotatE'] as KGEModelType[]).map(m => (
                    <button
                      key={m}
                      onClick={() => setModelType(m)}
                      className={`py-2.5 px-1.5 rounded-lg border transition-all cursor-pointer ${
                        modelType === m
                          ? 'bg-indigo-50 border-indigo-200 text-indigo-600 font-bold shadow-xs'
                          : 'bg-slate-50 border-slate-200/60 text-slate-500 hover:bg-slate-100/85 hover:text-slate-850'
                      }`}
                    >
                      {m === 'transE' && 'TransE (Translation)'}
                      {m === 'distMult' && 'DistMult (Bilinear)'}
                      {m === 'complEx' && 'ComplEx (Complex)'}
                      {m === 'rotatE' && 'RotatE (Rotation)'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Hyperparameter sliders */}
              <div className="flex flex-col gap-3.5 text-xs font-mono mt-1">
                {/* Embedding Dimension */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">EMBEDDING DIMENSION:</span>
                    <span className="text-slate-800 font-semibold">{dim}</span>
                  </div>
                  <input
                    type="range" min="10" max="100" step="5"
                    value={dim} onChange={(e) => setDim(Number(e.target.value))}
                    className="accent-indigo-600 bg-slate-100 h-1 rounded-lg cursor-pointer"
                  />
                  <span className="text-[9px] text-slate-500">
                    {modelType === 'complEx' ? `Effective real + imag parameters: ${dim * 2}` : `Embed size per node: ${dim}`}
                  </span>
                </div>

                {/* Learning Rate */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">LEARNING RATE (η):</span>
                    <span className="text-slate-800 font-semibold">{lr}</span>
                  </div>
                  <input
                    type="range" min="0.005" max="0.10" step="0.005"
                    value={lr} onChange={(e) => setLr(Number(e.target.value))}
                    className="accent-indigo-600 bg-slate-100 h-1 rounded-lg cursor-pointer"
                  />
                </div>

                {/* Margin */}
                {(modelType === 'transE' || modelType === 'rotatE') && (
                  <div className="flex flex-col gap-1">
                     <div className="flex justify-between">
                      <span className="text-slate-500">MARGIN (γ):</span>
                      <span className="text-slate-800 font-semibold">{margin}</span>
                    </div>
                    <input
                      type="range" min="1.0" max="8.0" step="0.5"
                      value={margin} onChange={(e) => setMargin(Number(e.target.value))}
                      className="accent-indigo-600 bg-slate-100 h-1 rounded-lg cursor-pointer"
                    />
                  </div>
                )}

                {/* Epochs */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">TRAINING EPOCHS:</span>
                    <span className="text-slate-800 font-semibold">{epochs}</span>
                  </div>
                  <input
                    type="range" min="50" max="500" step="10"
                    value={epochs} onChange={(e) => setEpochs(Number(e.target.value))}
                    className="accent-indigo-600 bg-slate-100 h-1 rounded-lg cursor-pointer"
                  />
                </div>

                {/* Neg samples */}
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">NEG SAMPLES PER TRIPLE:</span>
                    <span className="text-slate-800 font-semibold">{negSamples}</span>
                  </div>
                  <input
                    type="range" min="1" max="15" step="1"
                    value={negSamples} onChange={(e) => setNegSamples(Number(e.target.value))}
                    className="accent-indigo-600 bg-slate-100 h-1 rounded-lg cursor-pointer"
                  />
                </div>
              </div>

              {/* Train Button */}
              <button
                onClick={handleTrain}
                disabled={isTraining}
                className={`w-full py-3.5 rounded-xl font-bold font-mono text-xs tracking-[0.1em] uppercase transition-all flex items-center justify-center gap-2 cursor-pointer ${
                  isTraining
                    ? 'bg-slate-100 border border-slate-200 text-slate-400 cursor-not-allowed animate-pulse'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-500/10 hover:shadow-indigo-500/20 active:scale-[0.98]'
                }`}
              >
                {isTraining ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    TRAINING MODEL...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-current" />
                    GENERATE EMBEDDING MODEL
                  </>
                )}
              </button>

              {/* SVGLossCurve chart */}
              {lossHistory.length > 0 && (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                    <span>LEARNING LOSS CURVE</span>
                    <span className="text-indigo-600 font-bold">LOSS: {lossHistory[lossHistory.length-1].toFixed(5)}</span>
                  </div>
                  {renderSVGLossChart()}
                </div>
              )}
            </div>

          </div>

          {/* Right Panel: Interactive Visualizer & Analytics (col-span-8) */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            
            {/* 3D Projection Graph Canvas Card */}
            {currentProgress && isModelTrainedForSelectedConfig ? (
              <Graph3D
                entityEmbeddings={visualizerEntityEmbeddings || currentProgress.entityEmbeddings}
                triples={visualizerTriples}
                hoveredNode={hoveredNode}
                setHoveredNode={setHoveredNode}
                selectedNode={selectedNode}
                setSelectedNode={setSelectedNode}
                syntheticNames={syntheticConcepts.map(c => c.name)}
                nodeColors={nodeColors}
                projectionMode={projectionMode}
                setProjectionMode={setProjectionMode}
                clusteringMode={clusteringMode}
                setClusteringMode={setClusteringMode}
                numClusters={numClusters}
                setNumClusters={setNumClusters}
              />
            ) : (
              <div id="graph-placeholder-card" className="w-full h-[380px] md:h-[450px] rounded-2xl bg-white border border-slate-200/80 shadow-md shadow-slate-100 flex flex-col items-center justify-center text-center p-8 gap-4">
                <div className="relative p-4 rounded-full bg-slate-50 border border-slate-200 text-slate-400">
                  <Network className="h-10 w-10 text-indigo-500/50 animate-pulse" />
                  <span className="absolute inset-0 rounded-full border border-indigo-500/20 animate-ping"></span>
                </div>
                <div className="flex flex-col gap-1.5 max-w-sm">
                  <span className="text-xs font-bold tracking-[0.15em] font-mono text-slate-700 uppercase">
                    {engineInstance ? "MODEL TYPE CHANGED" : "EMBEDDING SPACE EMPTY"}
                  </span>
                  <p className="text-xs text-slate-500">
                    {engineInstance 
                      ? "The embedding model configuration has changed. Click Generate Embedding Model to train this formulation."
                      : "The 3D coordinate mapping needs a trained model. Adjust hyperparameters on the left and click Generate Embedding Model to watch the nodes self-organize."}
                  </p>
                </div>
              </div>
            )}

            {/* Deep Latent Space Exploration Workspace (Tabs Layout) */}
            <div id="explorer-tabs-container" className="rounded-2xl bg-white border border-slate-200/80 overflow-hidden shadow-md shadow-slate-100 flex flex-col">
              
              {/* Tab Navigation header */}
              <div className="w-full bg-slate-50/55 border-b border-slate-200/85 flex flex-wrap md:flex-nowrap font-mono text-[11px]">
                <button
                  onClick={() => setActiveTab('link')}
                  className={`flex-grow md:flex-none px-5 py-3.5 border-b-2 font-bold tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
                    activeTab === 'link' 
                      ? 'border-indigo-600 text-indigo-600 bg-white' 
                      : 'border-transparent text-slate-500 hover:text-slate-850 hover:bg-slate-100/50'
                  }`}
                >
                  <Search className="h-3.5 w-3.5" /> LINK PREDICTION
                </button>
                <button
                  onClick={() => setActiveTab('relations')}
                  className={`flex-grow md:flex-none px-5 py-3.5 border-b-2 font-bold tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
                    activeTab === 'relations' 
                      ? 'border-indigo-600 text-indigo-600 bg-white' 
                      : 'border-transparent text-slate-500 hover:text-slate-850 hover:bg-slate-100/50'
                  }`}
                >
                  <Sparkles className="h-3.5 w-3.5" /> HIDDEN RELATIONSHIPS
                </button>
                <button
                  onClick={() => setActiveTab('latent')}
                  className={`flex-grow md:flex-none px-5 py-3.5 border-b-2 font-bold tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
                    activeTab === 'latent' 
                      ? 'border-indigo-600 text-indigo-600 bg-white' 
                      : 'border-transparent text-slate-500 hover:text-slate-850 hover:bg-slate-100/50'
                  }`}
                >
                  <Cpu className="h-3.5 w-3.5" /> LATENT ENTITY GENERATOR
                </button>
                <button
                  onClick={() => setActiveTab('metrics')}
                  className={`flex-grow md:flex-none px-5 py-3.5 border-b-2 font-bold tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
                    activeTab === 'metrics' 
                      ? 'border-indigo-600 text-indigo-600 bg-white' 
                      : 'border-transparent text-slate-500 hover:text-slate-850 hover:bg-slate-100/50'
                  }`}
                >
                  <Activity className="h-3.5 w-3.5" /> METRICS & NEIGHBORS
                </button>
              </div>

              {/* Tab Workspace content */}
              <div className="p-5 flex flex-col gap-4 bg-white text-slate-700">
                {!isModelTrainedForSelectedConfig ? (
                  <div className="text-center py-16 font-mono text-xs text-slate-400 bg-transparent border border-dashed border-slate-200 rounded-xl flex flex-col items-center justify-center p-8 gap-4">
                    <div className="p-4 rounded-full bg-slate-50 border border-slate-150 text-slate-400">
                      <Cpu className="h-8 w-8 text-indigo-500/50 animate-pulse" />
                    </div>
                    <div className="flex flex-col gap-1.5 max-w-sm">
                      <span className="text-[10px] font-bold tracking-[0.15em] uppercase text-slate-600">
                        {engineInstance ? "Hyperparameters Changed" : "Untrained Formulation"}
                      </span>
                      <p className="text-slate-500 leading-relaxed text-[11.5px]">
                        {engineInstance 
                          ? `The embedding model is trained, but the active formulation (${modelType.toUpperCase()} d=${dim}) does not match the trained model in memory. Please train the current settings to enable predictions & diagnostics.`
                          : "The diagnostic workspace, link prediction suite, and latent arithmetic engine require a trained model. Configure the formulation parameters and click Generate Embedding Model."}
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* 1. Tab Link Prediction */}
                    {activeTab === 'link' && (
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-sm font-semibold text-slate-800">Triple Completer / Link Prediction</h3>
                      <p className="text-xs text-slate-500">
                        Query the high-dimensional geometry to complete a triple. Select a subject and relation, leaving the object empty <b className="text-indigo-600">(?, r, t)</b> or <b className="text-indigo-600">(h, r, ?)</b> to predict the top candidates.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
                      {/* Subject select */}
                      <div className="flex flex-col gap-1.5 font-mono text-xs">
                        <span className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">SUBJECT (h)</span>
                        <select
                          value={predSub}
                          onChange={(e) => setPredSub(e.target.value)}
                          className="bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-indigo-550 text-slate-700"
                        >
                          <option value="">[Any Subject / Predict (?)]</option>
                          {uniqueEntities.map(e => (
                            <option key={e} value={e}>{e}</option>
                          ))}
                        </select>
                      </div>

                      {/* Relation select */}
                      <div className="flex flex-col gap-1.5 font-mono text-xs">
                        <span className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">RELATION (r)</span>
                        <select
                          value={predRel}
                          onChange={(e) => setPredRel(e.target.value)}
                          className="bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-indigo-550 text-indigo-600 font-bold"
                        >
                          {uniqueRelations.map(r => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                      </div>

                      {/* Object select */}
                      <div className="flex flex-col gap-1.5 font-mono text-xs">
                        <span className="text-slate-400 font-bold uppercase tracking-wider text-[10px]">OBJECT (t)</span>
                        <select
                          value={predObj}
                          onChange={(e) => setPredObj(e.target.value)}
                          className="bg-slate-50 border border-slate-200 rounded-lg p-2 focus:outline-none focus:border-indigo-550 text-slate-700"
                        >
                          <option value="">[Predict (?)]</option>
                          {uniqueEntities.map(e => (
                            <option key={e} value={e}>{e}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="flex justify-end mt-1">
                      <button
                        onClick={handlePredict}
                        disabled={!engineInstance}
                        className={`px-6 py-2 rounded-xl text-xs font-mono font-bold tracking-wider transition-all uppercase cursor-pointer ${
                          engineInstance
                            ? 'bg-white hover:bg-slate-50 text-indigo-600 border border-slate-200 shadow-xs active:scale-95'
                            : 'bg-transparent border border-slate-100 text-slate-400 cursor-not-allowed'
                        }`}
                      >
                        RUN GEOMETRIC SCORING
                      </button>
                    </div>

                    {/* Results table */}
                    {predResults.length > 0 && (
                      <div className="mt-2 border border-slate-200 rounded-xl overflow-hidden font-mono text-xs">
                        <div className="bg-slate-50 text-slate-500 px-4 py-2 border-b border-slate-200 grid grid-cols-12 gap-2 text-[10px] tracking-wider uppercase font-bold">
                          <span className="col-span-6">PREDICTED TARGET CANDIDATE</span>
                          <span className="col-span-3 text-right">MODEL SCORE</span>
                          <span className="col-span-3 text-right">PROBABILITY %</span>
                        </div>
                        <div className="max-h-[220px] overflow-y-auto divide-y divide-slate-100 bg-white">
                          {predResults.map((res, i) => (
                            <div key={res.target} className="px-4 py-2.5 grid grid-cols-12 gap-2 items-center hover:bg-slate-50/50">
                              <span className="col-span-6 text-slate-800 font-bold flex items-center gap-2">
                                <span className="text-[10px] text-slate-400">#{i+1}</span>
                                {res.target}
                              </span>
                              <span className="col-span-3 text-right text-slate-500 text-[11px]">
                                {res.score.toFixed(4)}
                              </span>
                              <div className="col-span-3 flex items-center justify-end gap-2">
                                <span className="text-indigo-600 font-bold text-[11px]">
                                  {(res.probability * 100).toFixed(1)}%
                                </span>
                                <div className="w-12 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                                  <div className="h-full bg-gradient-to-r from-indigo-500 to-indigo-600" style={{ width: `${res.probability * 100}%` }} />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 2. Tab Hidden Relationships */}
                {activeTab === 'relations' && (
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-sm font-semibold text-slate-800">Latent Structure Discovery</h3>
                      <p className="text-xs text-slate-500">
                        These relationships are <b>not present in the training set</b> but score exceptionally high in the latent embedding space. These represent hidden truths or predictions made by the model!
                      </p>
                    </div>

                    {!engineInstance ? (
                      <div className="text-center py-8 font-mono text-xs text-slate-400 bg-transparent border border-dashed border-slate-200 rounded-xl">
                        Please train the embedding model to activate hidden relation discovery.
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {/* Discovery List */}
                        <div className="border border-slate-200 rounded-xl overflow-hidden flex flex-col font-mono text-xs">
                          <div className="bg-slate-50 text-slate-550 px-4 py-2.5 border-b border-slate-200 font-bold text-[10px] tracking-wider uppercase">
                            TOP 8 LATENT RELATIONSHIPS DISCOVERED
                          </div>
                          <div className="divide-y divide-slate-100 bg-white max-h-[300px] overflow-y-auto">
                            {discoveredLatentTriples.slice(0, 8).map((dt, i) => (
                              <div key={i} className="p-3 flex flex-col gap-2 hover:bg-slate-50/50">
                                <div className="flex items-center justify-between">
                                  <span className="text-[10px] text-indigo-600 font-bold flex items-center gap-1">
                                    <Sparkles className="h-3 w-3" /> CONFIDENCE: {(dt.confidence * 100).toFixed(1)}%
                                  </span>
                                  <span className="text-[10px] text-slate-400">Score: {dt.score.toFixed(3)}</span>
                                </div>
                                <div className="flex items-center flex-wrap gap-1 text-[11px]">
                                  <span className="bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-slate-800 font-semibold">{dt.sub}</span>
                                  <span className="text-indigo-600 font-bold">➔ {dt.rel} ➔</span>
                                  <span className="bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-slate-800 font-semibold">{dt.obj}</span>
                                </div>
                                <button
                                  onClick={() => handleExplainRelation(dt.sub, dt.rel, dt.obj, dt.score, dt.confidence)}
                                  className="self-end text-[10px] text-indigo-600 hover:text-indigo-700 font-semibold uppercase flex items-center gap-1 cursor-pointer"
                                >
                                  EXPLAIN WITH GEMINI ➔
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Gemini Explanation Detail view */}
                        <div className="border border-slate-200 rounded-xl overflow-hidden bg-slate-50/30 flex flex-col min-h-[300px]">
                          <div className="bg-slate-50 text-slate-550 px-4 py-2.5 border-b border-slate-200 font-mono text-xs font-bold flex items-center justify-between text-[10px] tracking-wider uppercase">
                            <span>GEMINI LATENT EXPLANATION</span>
                            {isGeneratingExplanation && <span className="animate-pulse text-indigo-600">ANALYZING...</span>}
                          </div>
                          
                          <div className="p-4 flex-grow flex flex-col gap-2 text-xs">
                            {selectedTripleExplanation ? (
                              <div className="flex flex-col gap-3">
                                <div className="p-2.5 rounded bg-white border border-slate-200 font-mono text-[11px]">
                                  <span className="text-[9px] text-slate-450 block font-semibold">EXPLAINING RELATION</span>
                                  <b className="text-slate-800">({selectedTripleExplanation.sub}, {selectedTripleExplanation.rel}, {selectedTripleExplanation.obj})</b>
                                </div>
                                <div className="text-slate-650 leading-relaxed max-h-[220px] overflow-y-auto pr-1">
                                  <div className="whitespace-pre-line text-slate-650 font-sans text-xs">
                                    {selectedTripleExplanation.text}
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 font-mono text-xs p-4 gap-2">
                                <Info className="h-6 w-6 text-slate-350" />
                                Select a latent relationship on the left and click "Explain with Gemini" to get AI insight on what this hidden relationship represents.
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 3. Tab Latent Space Entity Creator */}
                {activeTab === 'latent' && (
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-sm font-semibold text-slate-800">Synthesize Unconfirmed Entities (Latent Spaces)</h3>
                      <p className="text-xs text-slate-500">
                        Create a brand-new, hypothetical entity by algebraically combining vectors in the latent field! e.g., <b>h1 + h2 - h3</b>. We project this synthetic vector onto relations to describe its properties.
                      </p>
                    </div>

                    {!engineInstance ? (
                      <div className="text-center py-8 font-mono text-xs text-slate-400 bg-transparent border border-dashed border-slate-200 rounded-xl">
                        Please train the embedding model to activate the latent space entity generator.
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 font-mono text-xs">
                        
                        {/* Setup form */}
                        <div className="lg:col-span-5 flex flex-col gap-4">
                          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col gap-3.5">
                            <span className="font-bold text-slate-500 block text-[10px] tracking-wider uppercase">HYPOTHETICAL CONCEPT SETUP</span>
                            
                            <div className="flex flex-col gap-1">
                              <label className="text-[10px] text-slate-450 font-semibold">SYNTHETIC CONCEPT NAME</label>
                              <input
                                type="text"
                                value={latentName}
                                onChange={(e) => setLatentName(e.target.value.replace(/\s+/g, '_'))}
                                className="w-full bg-white border border-slate-200 rounded px-2.5 py-1.5 text-slate-800 focus:outline-none focus:border-indigo-500"
                                placeholder="e.g. Athena_Warfare"
                              />
                            </div>

                            {/* Arithmetic inputs */}
                            <div className="flex flex-col gap-2">
                              <label className="text-[10px] text-slate-450 font-semibold">VECTOR ALGEBRA DEFINITION</label>
                              
                              {/* Entity 1 */}
                              <select
                                value={latentSources[0]}
                                onChange={(e) => {
                                  const copy = [...latentSources];
                                  copy[0] = e.target.value;
                                  setLatentSources(copy);
                                }}
                                className="bg-white border border-slate-200 rounded p-1.5 text-slate-700 focus:outline-none font-mono text-[11px]"
                              >
                                <option value="">[None]</option>
                                {uniqueEntities.map(e => <option key={e} value={e}>{e}</option>)}
                              </select>

                              {/* Op 1 & Entity 2 */}
                              <div className="flex gap-2 items-center">
                                <select
                                  value={latentOps[0]}
                                  onChange={(e) => {
                                    const copy = [...latentOps];
                                    copy[0] = e.target.value as any;
                                    setLatentOps(copy);
                                  }}
                                  className="bg-white border border-slate-200 rounded p-1 text-indigo-600 font-bold focus:outline-none"
                                >
                                  <option value="add">+</option>
                                  <option value="sub">-</option>
                                </select>
                                <select
                                  value={latentSources[1]}
                                  onChange={(e) => {
                                    const copy = [...latentSources];
                                    copy[1] = e.target.value;
                                    setLatentSources(copy);
                                  }}
                                  className="flex-grow bg-white border border-slate-200 rounded p-1.5 text-slate-700 focus:outline-none font-mono text-[11px]"
                                >
                                  <option value="">[None]</option>
                                  {uniqueEntities.map(e => <option key={e} value={e}>{e}</option>)}
                                </select>
                              </div>

                              {/* Op 2 & Entity 3 */}
                              <div className="flex gap-2 items-center">
                                <select
                                  value={latentOps[1] || 'add'}
                                  onChange={(e) => {
                                    const copy = [...latentOps];
                                    copy[1] = e.target.value as any;
                                    setLatentOps(copy);
                                  }}
                                  className="bg-white border border-slate-200 rounded p-1 text-indigo-600 font-bold focus:outline-none"
                                >
                                  <option value="add">+</option>
                                  <option value="sub">-</option>
                                </select>
                                <select
                                  value={latentSources[2] || ''}
                                  onChange={(e) => {
                                    const copy = [...latentSources];
                                    copy[2] = e.target.value;
                                    setLatentSources(copy);
                                  }}
                                  className="flex-grow bg-white border border-slate-200 rounded p-1.5 text-slate-700 focus:outline-none font-mono text-[11px]"
                                >
                                  <option value="">[None]</option>
                                  {uniqueEntities.map(e => <option key={e} value={e}>{e}</option>)}
                                </select>
                              </div>
                            </div>

                            <div className="flex gap-2 mt-2">
                              <button
                                onClick={handleCalculateLatentEntity}
                                className="flex-grow py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 rounded font-bold uppercase transition-all cursor-pointer text-[11px] shadow-xs"
                              >
                                CALCULATE RELATIONS
                              </button>
                              {Object.keys(latentRelations).length > 0 && (
                                <button
                                  onClick={handleExplainLatentEntity}
                                  disabled={isGeneratingLatentConcept}
                                  className="px-3 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded font-bold uppercase flex items-center justify-center gap-1 transition-all cursor-pointer text-[11px]"
                                  title="Explain Concept with Gemini"
                                >
                                  <Sparkles className="h-4 w-4" /> EXPLAIN
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Active Synthetic Concepts card */}
                          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col gap-3">
                            <span className="font-bold text-slate-500 block text-[10px] tracking-wider uppercase flex items-center justify-between">
                              <span>ACTIVE SYNTHETIC CONCEPTS</span>
                              <span className="text-[10px] font-mono text-indigo-600 font-bold bg-indigo-50 px-1.5 py-0.5 rounded">{syntheticConcepts.length}</span>
                            </span>
                            {syntheticConcepts.length === 0 ? (
                              <p className="text-[10px] text-slate-400 italic font-mono leading-relaxed">No synthetic concepts generated yet. Configure options above and click CALCULATE RELATIONS to add.</p>
                            ) : (
                              <div className="flex flex-col gap-1.5 max-h-[220px] overflow-y-auto pr-1">
                                {syntheticConcepts.map(sc => (
                                  <div key={sc.name} className="flex items-center justify-between p-2.5 bg-white border border-slate-200 rounded-lg hover:border-slate-350 transition-all">
                                    <div className="flex flex-col gap-0.5 min-w-0">
                                      <span className="font-bold text-slate-800 text-[11px] truncate">{sc.name}</span>
                                      <span className="text-[9px] text-slate-450 truncate font-mono">
                                        {sc.sourceEntities[0]} {sc.sourceEntities.slice(1).map((se, i) => `${sc.ops[i] === 'add' ? '+' : '-'} ${se}`).join(' ')}
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-1.5 shrink-0">
                                      <button
                                        onClick={() => {
                                          setLatentName(sc.name);
                                          const fullSources = ['', '', ''];
                                          sc.sourceEntities.forEach((se, idx) => {
                                            if (idx < 3) fullSources[idx] = se;
                                          });
                                          setLatentSources(fullSources);
                                          const fullOps = ['add', 'sub'];
                                          sc.ops.forEach((op, idx) => {
                                            if (idx < 2) fullOps[idx] = op;
                                          });
                                          setLatentOps(fullOps as any);
                                          
                                          if (engineInstance) {
                                            const preds = engineInstance.generateLatentEntity(
                                              sc.name,
                                              sc.sourceEntities,
                                              sc.ops,
                                              modelType,
                                              dim
                                            );
                                            setLatentRelations(preds);
                                            setLatentAnalysis('');
                                          }
                                        }}
                                        className="px-2 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 text-[9px] font-bold rounded uppercase transition-all cursor-pointer"
                                        title="Load into settings"
                                      >
                                        LOAD
                                      </button>
                                      <button
                                        onClick={() => handleDeleteSyntheticConcept(sc.name)}
                                        className="p-1 hover:bg-rose-50 text-rose-500 hover:text-rose-600 rounded transition-all cursor-pointer"
                                        title="Delete concept"
                                      >
                                        <Trash2 className="h-3.5 w-3.5" />
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Predictions & Gemini analysis */}
                        <div className="lg:col-span-7 flex flex-col gap-4">
                          
                          {/* Relations projection */}
                          {Object.keys(latentRelations).length > 0 && (() => {
                            const sortedLatentRelations = Object.keys(latentRelations)
                              .map(rel => {
                                const topObj = latentRelations[rel]?.[0];
                                return { rel, topObj };
                              })
                              .filter((item): item is { rel: string; topObj: { target: string; score: number } } => item.topObj !== undefined)
                              .sort((a, b) => {
                                if (latentSortOrder === 'alphabetical') {
                                  return a.rel.localeCompare(b.rel);
                                }
                                if (latentSortOrder === 'desc') {
                                  return b.topObj.score - a.topObj.score;
                                } else {
                                  return a.topObj.score - b.topObj.score;
                                }
                              });

                            const isDistanceModel = modelType === 'transE' || modelType === 'rotatE';

                            return (
                              <div className="flex flex-col gap-3.5">
                                {/* Sorting & Note Panel */}
                                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 flex flex-col gap-2.5">
                                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
                                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono">
                                      Sort Predicted Relationships
                                    </span>
                                    <div className="flex rounded-lg bg-slate-200/60 p-0.5 border border-slate-200 shadow-xs">
                                      <button
                                        onClick={() => setLatentSortOrder('desc')}
                                        className={`px-2.5 py-1 rounded-md text-[9px] font-bold font-mono transition-all cursor-pointer ${
                                          latentSortOrder === 'desc'
                                            ? 'bg-white text-indigo-700 shadow-xs'
                                            : 'text-slate-500 hover:text-slate-700'
                                        }`}
                                      >
                                        MOST PLAUSIBLE
                                      </button>
                                      <button
                                        onClick={() => setLatentSortOrder('asc')}
                                        className={`px-2.5 py-1 rounded-md text-[9px] font-bold font-mono transition-all cursor-pointer ${
                                          latentSortOrder === 'asc'
                                            ? 'bg-white text-indigo-700 shadow-xs'
                                            : 'text-slate-500 hover:text-slate-700'
                                        }`}
                                      >
                                        LEAST PLAUSIBLE
                                      </button>
                                      <button
                                        onClick={() => setLatentSortOrder('alphabetical')}
                                        className={`px-2.5 py-1 rounded-md text-[9px] font-bold font-mono transition-all cursor-pointer ${
                                          latentSortOrder === 'alphabetical'
                                            ? 'bg-white text-indigo-700 shadow-xs'
                                            : 'text-slate-500 hover:text-slate-700'
                                        }`}
                                      >
                                        A-Z RELATION
                                      </button>
                                    </div>
                                  </div>

                                  <div className="text-[10px] text-slate-500 leading-relaxed font-sans bg-white border border-slate-100 p-2.5 rounded-lg flex gap-2 items-start shadow-xs">
                                    <div className="p-1 rounded-md bg-indigo-50 text-indigo-600 mt-0.5 shrink-0">
                                      <Sparkles className="h-3.5 w-3.5" />
                                    </div>
                                    <div>
                                      <span className="font-bold text-slate-700">Model formulation note: </span>
                                      {isDistanceModel ? (
                                        <span>
                                          The active model (<strong className="text-indigo-650">{modelType.toUpperCase()}</strong>) is distance-based. Plausibility scores are <strong>negative distances</strong>; scores closer to zero (e.g. <code className="bg-slate-100 px-1 py-0.5 rounded text-indigo-600 font-mono font-bold text-[9px]">-0.05</code> vs <code className="bg-slate-100 px-1 py-0.5 rounded text-indigo-600 font-mono font-bold text-[9px]">-4.20</code>) represent a shorter translation gap and hence a more plausible match.
                                        </span>
                                      ) : (
                                        <span>
                                          The active model (<strong className="text-indigo-650">{modelType.toUpperCase()}</strong>) is similarity-based. Plausibility scores are <strong>bilinear products</strong>; higher, more positive scores represent a stronger correlation / more plausible match.
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                </div>

                                {/* Relationships Table */}
                                <div className="border border-slate-200 rounded-xl overflow-hidden max-h-[300px] overflow-y-auto bg-white shadow-inner scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent">
                                  <div className="bg-slate-50 px-3 py-2.5 border-b border-slate-200 text-[10px] font-bold text-slate-550 tracking-wider uppercase sticky top-0 z-10 flex items-center justify-between">
                                    <span>TOP TARGET PER RELATION FOR "{latentName}" ({sortedLatentRelations.length})</span>
                                    <span className="text-[9px] text-indigo-600 font-bold bg-indigo-50 px-1.5 py-0.5 rounded uppercase font-mono">SCROLLABLE</span>
                                  </div>
                                  <div className="p-3 divide-y divide-slate-100 bg-white">
                                    {sortedLatentRelations.map(({ rel, topObj }) => (
                                      <div key={rel} className="py-2.5 flex justify-between items-center text-[11px] hover:bg-slate-50/50 px-1 rounded-lg transition-colors">
                                        <div className="flex items-center gap-1">
                                          <span className="text-slate-400 font-mono text-[9px]">➔</span>
                                          <b className="text-indigo-600 font-mono">{rel}</b>
                                          <span className="text-slate-400 font-mono text-[9px]">➔</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <span className="text-slate-800 font-bold">{topObj.target}</span>
                                          <span className="text-[9px] font-mono text-slate-400 bg-slate-50 border border-slate-100 px-1.5 py-0.5 rounded">
                                            {topObj.score.toFixed(3)}
                                          </span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}

                          {/* Gemini analysis text */}
                          {latentAnalysis && (
                            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 max-h-[220px] overflow-y-auto">
                              <span className="text-[10px] text-indigo-600 font-bold block mb-2 flex items-center gap-1 tracking-wider uppercase">
                                <Sparkles className="h-3.5 w-3.5" /> GEMINI CONCEPTUAL SYNTHESIS
                              </span>
                              <div className="text-[11px] whitespace-pre-line text-slate-650 font-sans leading-relaxed">
                                {latentAnalysis}
                              </div>
                            </div>
                          )}

                        </div>

                      </div>
                    )}
                  </div>
                )}

                {/* 4. Tab metrics & Neighbors */}
                {activeTab === 'metrics' && (
                  <div className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1">
                      <h3 className="text-sm font-semibold text-slate-800">Diagnostics, Dimensionality & Semantic Closeness</h3>
                      <p className="text-xs text-slate-500">
                        Check semantic neighbors for specific entities based on cosine similarity, representing clusters formed in the high-dimensional embedding space.
                      </p>
                    </div>

                    {!engineInstance ? (
                      <div className="text-center py-8 font-mono text-xs text-slate-400 bg-transparent border border-dashed border-slate-200 rounded-xl">
                        Please train the embedding model to activate diagnostic metrics.
                      </div>
                    ) : isTraining ? (
                      <div className="text-center py-12 font-mono text-xs text-slate-500 bg-slate-50 border border-dashed border-slate-200 rounded-xl flex flex-col items-center justify-center gap-3">
                        <RefreshCw className="h-5 w-5 text-indigo-500 animate-spin" />
                        <span className="font-semibold text-slate-700">Generating embedding representations...</span>
                        <span className="text-[10px] text-slate-400">Diagnostics, dimensionality & semantic closeness will be computed once the model finishes generating.</span>
                      </div>
                    ) : (
                      <>
                        {/* KGE Model Quality Evaluation Dashboard */}
                        {kgeMetrics && (
                          <div className="flex flex-col gap-2.5 p-4 rounded-2xl bg-indigo-50/40 border border-indigo-100/80">
                            <span className="font-bold text-indigo-600 text-[10px] tracking-wider uppercase">LINK PREDICTION QUALITY METRICS</span>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                              <div className="p-3 bg-white border border-indigo-100/60 rounded-xl shadow-xs flex flex-col items-center justify-center text-center">
                                <span className="text-[9px] font-bold text-slate-450 uppercase tracking-wider block mb-1">Mean Reciprocal Rank (MRR)</span>
                                <span className="text-base font-bold text-indigo-600 font-mono">{kgeMetrics.mrr.toFixed(4)}</span>
                              </div>
                              <div className="p-3 bg-white border border-indigo-100/60 rounded-xl shadow-xs flex flex-col items-center justify-center text-center">
                                <span className="text-[9px] font-bold text-slate-450 uppercase tracking-wider block mb-1">Hits @ 1</span>
                                <span className="text-base font-bold text-slate-700 font-mono">{(kgeMetrics.hits1 * 100).toFixed(1)}%</span>
                              </div>
                              <div className="p-3 bg-white border border-indigo-100/60 rounded-xl shadow-xs flex flex-col items-center justify-center text-center">
                                <span className="text-[9px] font-bold text-slate-450 uppercase tracking-wider block mb-1">Hits @ 5</span>
                                <span className="text-base font-bold text-slate-700 font-mono">{(kgeMetrics.hits5 * 100).toFixed(1)}%</span>
                              </div>
                              <div className="p-3 bg-white border border-indigo-100/60 rounded-xl shadow-xs flex flex-col items-center justify-center text-center">
                                <span className="text-[9px] font-bold text-slate-450 uppercase tracking-wider block mb-1">Hits @ 10</span>
                                <span className="text-base font-bold text-slate-700 font-mono">{(kgeMetrics.hits10 * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 font-mono text-xs">
                          {/* Local neighbors lookup */}
                          <div className="border border-slate-200 rounded-xl bg-slate-50/30 p-4 flex flex-col gap-3">
                            <span className="font-bold text-slate-500 text-[10px] tracking-wider uppercase">SEMANTIC NEAREST NEIGHBORS</span>
                          <div className="flex flex-col gap-1">
                            <label className="text-[10px] text-slate-450">CHOOSE ENTITY</label>
                            <select
                              value={selectedNode || ''}
                              onChange={(e) => setSelectedNode(e.target.value || null)}
                              className="bg-white border border-slate-200 rounded p-2 text-slate-800 focus:outline-none"
                            >
                              <option value="">[Select an entity to explore neighbors]</option>
                              {uniqueEntities.map(e => <option key={e} value={e}>{e}</option>)}
                            </select>
                          </div>

                          {selectedNode ? (
                            <div className="flex flex-col gap-2 mt-2">
                              <span className="text-[10px] text-slate-450 font-semibold">EMBEDDING SLICE:</span>
                              <span className="text-[10px] text-indigo-600 bg-white border border-slate-200 p-2 rounded truncate block">{getEmbeddingsValueStr(selectedNode)}</span>
                              
                              <div className="border border-slate-200 rounded-lg overflow-hidden mt-1 divide-y divide-slate-100 bg-white">
                                {nearestNeighbors.map((n, i) => (
                                  <div key={n.entity} className="p-2 flex justify-between items-center text-[11px] bg-white">
                                    <span>#{i+1} <b className="text-slate-700">{n.entity}</b></span>
                                    <span className="text-indigo-600 font-bold">{(n.similarity * 100).toFixed(1)}% similarity</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ) : (
                            <p className="text-slate-400 text-[10px] text-center py-4">
                              Select a node on the 3D graph or choose from the list above to explore semantic similarity clusters.
                            </p>
                          )}
                        </div>

                        {/* Global analysis summary with Gemini */}
                        <div className="border border-slate-200 bg-slate-50/30 rounded-xl p-4 flex flex-col gap-3">
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-slate-500 text-[10px] tracking-wider uppercase">GLOBAL TOPOLOGY ASSESSMENT</span>
                            <button
                              onClick={handleGlobalAnalysis}
                              disabled={isGeneratingGlobalAnalysis}
                              className="px-3 py-1 bg-white border border-slate-200 hover:bg-slate-50 rounded text-[10px] font-bold text-indigo-600 shadow-xs transition-all uppercase cursor-pointer"
                            >
                              COMPILE WITH GEMINI
                            </button>
                          </div>

                          <div className="p-3.5 rounded-lg bg-white border border-slate-200 h-full max-h-[220px] overflow-y-auto">
                            {globalAnalysisText ? (
                              <div className="whitespace-pre-line text-[11px] font-sans text-slate-650 leading-relaxed">
                                {globalAnalysisText}
                              </div>
                            ) : (
                              <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 text-[10px] py-4 gap-2">
                                <FileText className="h-6 w-6 text-slate-350" />
                                Click "Compile with Gemini" to analyze structural metrics, convergence loss, and relational traits of the trained KGE space.
                              </div>
                            )}
                          </div>
                        </div>

                      </div>

                      {/* Out-of-Distribution Data Hygiene Panel */}
                      <div className="border border-slate-200 bg-white rounded-xl p-5 mt-4 flex flex-col gap-4 shadow-sm text-xs font-sans">
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
                          <div className="flex items-center gap-2.5">
                            <div className="p-2 rounded-lg bg-rose-50 border border-rose-100 text-rose-600">
                              <ShieldAlert className="h-4.5 w-4.5" />
                            </div>
                            <div className="flex flex-col">
                              <span className="font-bold text-slate-800 text-[11px] tracking-wider uppercase">DATA HYGIENE & OOD ANOMALY DETECTOR</span>
                              <span className="text-[10px] text-slate-450 font-mono">
                                {modelType === 'rotatE' ? (
                                  <span>{"Metric: d_r(h, t) = ||h ∘ r - t|| (Complex Phase Rotations)"}</span>
                                ) : modelType === 'transE' ? (
                                  <span>{"Metric: d_r(h, t) = ||h + r - t|| (Translational Shifts)"}</span>
                                ) : modelType === 'complEx' ? (
                                  <span>{"Metric: d_r(h, t) = ||h ∘ r - t|| (Hermitian Multiplication)"}</span>
                                ) : (
                                  <span>{"Metric: d_r(h, t) = ||h ∘ r - t|| (Bilinear Reconstruction)"}</span>
                                )}
                              </span>
                            </div>
                          </div>

                          {/* Filter and Search controls */}
                          <div className="flex flex-wrap items-center gap-2">
                            {/* Search */}
                            <div className="relative">
                              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-slate-400" />
                              <input
                                type="text"
                                placeholder="Search entities..."
                                value={hygieneSearch}
                                onChange={(e) => setHygieneSearch(e.target.value)}
                                className="pl-8 pr-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-[11px] font-mono focus:outline-none focus:border-indigo-500 w-[160px] md:w-[200px]"
                              />
                            </div>

                            {/* Filter Segmented Control */}
                            <div className="flex border border-slate-200 rounded-lg overflow-hidden bg-slate-50 p-0.5 text-[10px] font-mono font-bold">
                              <button
                                onClick={() => setHygieneFilter('anomalies')}
                                className={`px-3 py-1 rounded-md transition-all cursor-pointer ${
                                  hygieneFilter === 'anomalies'
                                    ? 'bg-white text-rose-600 shadow-xs'
                                    : 'text-slate-500 hover:text-slate-800'
                                }`}
                              >
                                Anomalies Only
                              </button>
                              <button
                                onClick={() => setHygieneFilter('all')}
                                className={`px-3 py-1 rounded-md transition-all cursor-pointer ${
                                  hygieneFilter === 'all'
                                    ? 'bg-white text-slate-750 shadow-xs'
                                    : 'text-slate-500 hover:text-slate-800'
                                }`}
                              >
                                All Triples
                              </button>
                            </div>
                          </div>
                        </div>

                        {/* Explanation banner */}
                        <div className="p-3 rounded-lg bg-slate-50 border border-slate-150/85 text-[11px] text-slate-650 leading-relaxed">
                          <span className="font-bold text-slate-800 mr-1 flex items-center gap-1">
                            <Info className="h-3.5 w-3.5 text-indigo-500 inline" /> Mathematical Hypothesis:
                          </span>
                          {"Anomalies represent triples that fail to satisfy the learned geometric/algebraic transitions of the model. If a trained relation represents rotation (RotatE) or translation (TransE), a high violation distance (d >> mean) implies the facts are incompatible with the global embedding regularities."}
                        </div>

                        {/* Triple table list */}
                        <div className="border border-slate-150 rounded-lg overflow-hidden max-h-[300px] overflow-y-auto divide-y divide-slate-100 bg-white">
                          {(() => {
                            if (isTraining) {
                              return (
                                <div className="p-12 text-center text-[11px] text-slate-500 font-mono flex flex-col items-center justify-center gap-2">
                                  <RefreshCw className="h-4.5 w-4.5 text-rose-500 animate-spin animate-spin-reverse" />
                                  <span className="font-semibold text-slate-750">Learning rotational transitions & vector alignments...</span>
                                  <span className="text-[10px] text-slate-400">Out-of-distribution anomaly detection will be completed once the model finishes generating.</span>
                                </div>
                              );
                            }

                            const filtered = oodAnomalies.filter(item => {
                              const matchesSearch = 
                                item.triple.sub.toLowerCase().includes(hygieneSearch.toLowerCase()) ||
                                item.triple.rel.toLowerCase().includes(hygieneSearch.toLowerCase()) ||
                                item.triple.obj.toLowerCase().includes(hygieneSearch.toLowerCase());
                              
                              if (hygieneFilter === 'anomalies') {
                                return matchesSearch && item.isAnomaly;
                              }
                              return matchesSearch;
                            });

                            if (filtered.length === 0) {
                              return (
                                <div className="p-8 text-center text-[11px] text-slate-400 font-mono">
                                  No triples matching the criteria. Dataset hygiene is pristine!
                                </div>
                              );
                            }

                            return filtered.map((item, index) => {
                              const maxAllowedDist = Math.max(...oodAnomalies.map(a => a.distance)) || 1.0;
                              const barWidth = Math.min(100, (item.distance / maxAllowedDist) * 100);
                              
                              return (
                                <div 
                                  key={index} 
                                  className="p-3 hover:bg-slate-50/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3 transition-colors text-[11px] font-mono group"
                                >
                                  <div className="flex flex-col gap-1.5 flex-grow">
                                    <div className="flex items-center flex-wrap gap-1 text-[11px]">
                                      <span className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-700 font-bold border border-slate-200/50">{item.triple.sub}</span>
                                      <span className="text-indigo-600 font-bold text-[10px] mx-1">-{item.triple.rel}→</span>
                                      <span className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-700 font-bold border border-slate-200/50">{item.triple.obj}</span>
                                    </div>

                                    <div className="flex items-center gap-4 text-[10px] text-slate-450">
                                      <span>Distance: <b className="text-slate-650 font-bold">{item.distance.toFixed(4)}</b></span>
                                      <span>z-Score: <b className={`font-bold ${item.isAnomaly ? 'text-rose-600' : 'text-slate-600'}`}>{item.zScore > 0 ? '+' : ''}{item.zScore.toFixed(2)}</b></span>
                                      
                                      <div className="w-20 bg-slate-100 h-1.5 rounded-full overflow-hidden hidden sm:block">
                                        <div 
                                          className={`h-full rounded-full transition-all duration-500 ${
                                            item.isAnomaly ? 'bg-rose-500' : item.zScore > 0.4 ? 'bg-amber-400' : 'bg-emerald-500'
                                          }`}
                                          style={{ width: `${barWidth}%` }}
                                        />
                                      </div>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-2 shrink-0 self-end sm:self-auto">
                                    {item.isAnomaly ? (
                                      <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-rose-50 text-rose-600 border border-rose-100 uppercase tracking-wider flex items-center gap-1">
                                        <AlertTriangle className="h-3 w-3" /> Anomaly Outlier
                                      </span>
                                    ) : item.zScore > 0.4 ? (
                                      <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-amber-50 text-amber-600 border border-amber-100 uppercase tracking-wider">
                                        High Variance
                                      </span>
                                    ) : (
                                      <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-emerald-50 text-emerald-600 border border-emerald-150 uppercase tracking-wider">
                                        Coherent
                                      </span>
                                    )}

                                    <button
                                      onClick={() => {
                                        setPredSub(item.triple.sub);
                                        setPredRel(item.triple.rel);
                                        setPredObj(item.triple.obj);
                                        setActiveTab('link');
                                        document.getElementById('explorer-tabs-container')?.scrollIntoView({ behavior: 'smooth' });
                                      }}
                                      className="opacity-0 group-hover:opacity-100 px-2.5 py-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 rounded text-[9px] font-bold transition-all uppercase tracking-wider flex items-center gap-1 cursor-pointer border border-indigo-100"
                                    >
                                      Debug
                                    </button>
                                  </div>
                                </div>
                              );
                            });
                          })()}
                        </div>
                      </div>
                    </>
                    )}

                  </div>
                )}

                  </>
                )}
              </div>

            </div>

          </div>

        </div>
      </main>

      {/* Footer */}
      <footer id="app-footer" className="w-full py-6 px-6 border-t border-slate-200 mt-auto bg-white shadow-xs">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between text-[11px] font-mono text-slate-500 gap-4">
          <span>KNOWLEDGE GRAPH EMBEDDINGS (KGE) INTERACTIVE LAB</span>
          <span className="flex items-center gap-1 uppercase">
            POWERED BY <b className="text-slate-600">TS-MATRIXAutograd</b> & <b className="text-indigo-600">GEMINI 2.5 FLASH</b>
          </span>
        </div>
      </footer>

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-fade-in">
          <div className="w-full max-w-md bg-white rounded-2xl border border-slate-200 shadow-2xl p-6 flex flex-col gap-5 relative">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Settings className="h-5 w-5 text-indigo-600 animate-spin-slow" />
                <h3 className="text-sm font-bold text-slate-950 uppercase font-mono">LLM Configuration Settings</h3>
              </div>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="p-1 rounded-md hover:bg-slate-50 text-slate-400 hover:text-slate-600 transition-colors cursor-pointer"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Provider selection */}
            <div className="flex flex-col gap-2">
              <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">Select Model Provider</label>
              <div className="grid grid-cols-3 gap-2">
                {(['gemini', 'ollama', 'openai'] as const).map((prov) => (
                  <button
                    key={prov}
                    onClick={() => setLlmConfig(prev => ({ ...prev, provider: prov }))}
                    className={`px-3 py-2.5 rounded-xl border text-xs font-mono uppercase tracking-wider font-bold flex flex-col items-center justify-center gap-1 transition-all cursor-pointer ${
                      llmConfig.provider === prov
                        ? 'border-indigo-600 bg-indigo-50/50 text-indigo-750'
                        : 'border-slate-200 bg-white hover:bg-slate-50 text-slate-600 hover:text-slate-800'
                    }`}
                  >
                    <span>{prov === 'gemini' ? 'Gemini' : prov === 'ollama' ? 'Ollama' : 'OpenAI'}</span>
                    <span className="text-[8px] font-normal text-slate-400 uppercase">
                      {prov === 'gemini' ? 'Google' : prov === 'ollama' ? 'Local LLM' : 'SaaS API'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Provider Configuration */}
            {llmConfig.provider === 'gemini' && (
              <div className="flex flex-col gap-3.5 bg-slate-50/85 p-4 rounded-xl border border-slate-100">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">Custom Gemini API Key (Optional)</label>
                  <input
                    type="password"
                    placeholder="Leave empty to use pre-configured server key"
                    value={llmConfig.geminiApiKey}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, geminiApiKey: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                  <p className="text-[9px] text-slate-400 leading-relaxed">
                    By default, this app routes requests through our server-side Gemini API using secure developer keys. Fill this in only if you wish to use your personal API key.
                  </p>
                </div>
              </div>
            )}

            {llmConfig.provider === 'ollama' && (
              <div className="flex flex-col gap-3.5 bg-slate-50/85 p-4 rounded-xl border border-slate-100">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">Ollama API Endpoint URL</label>
                  <input
                    type="text"
                    placeholder="e.g. http://localhost:11434"
                    value={llmConfig.ollamaEndpoint}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, ollamaEndpoint: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">Local Model Name</label>
                  <input
                    type="text"
                    placeholder="e.g. gemma2, llama3, mistral"
                    value={llmConfig.ollamaModel}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, ollamaModel: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                  <p className="text-[9px] text-slate-400 leading-relaxed">
                    Ensure Ollama is running locally with CORS enabled (`OLLAMA_ORIGINS="*"`) and model is pulled.
                  </p>
                </div>
              </div>
            )}

            {llmConfig.provider === 'openai' && (
              <div className="flex flex-col gap-3.5 bg-slate-50/85 p-4 rounded-xl border border-slate-100">
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">OpenAI API Endpoint URL</label>
                  <input
                    type="text"
                    placeholder="e.g. https://api.openai.com/v1"
                    value={llmConfig.openaiEndpoint}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, openaiEndpoint: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">OpenAI API Key</label>
                  <input
                    type="password"
                    placeholder="sk-..."
                    value={llmConfig.openaiApiKey}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, openaiApiKey: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <label className="text-[10px] uppercase tracking-wider text-slate-400 font-mono font-bold">Model Name</label>
                  <input
                    type="text"
                    placeholder="e.g. gpt-4o-mini"
                    value={llmConfig.openaiModel}
                    onChange={(e) => setLlmConfig(prev => ({ ...prev, openaiModel: e.target.value }))}
                    className="w-full bg-white border border-slate-250 rounded-lg px-3 py-2 text-xs text-slate-800 font-mono focus:outline-none focus:border-indigo-500/40"
                  />
                  <p className="text-[9px] text-slate-400 leading-relaxed">
                    Can also be used with OpenAI-compatible local endpoints like LM Studio or LocalAI (e.g., using `http://localhost:1234/v1`).
                  </p>
                </div>
              </div>
            )}

            <div className="flex gap-2.5 mt-2">
              <button
                onClick={() => {
                  // Reset config
                  setLlmConfig({
                    provider: 'gemini',
                    geminiApiKey: '',
                    ollamaEndpoint: 'http://localhost:11434',
                    ollamaModel: 'gemma2',
                    openaiApiKey: '',
                    openaiEndpoint: 'https://api.openai.com/v1',
                    openaiModel: 'gpt-4o-mini',
                  });
                }}
                className="px-3.5 py-2.5 rounded-xl border border-slate-200 hover:bg-slate-50 text-slate-500 hover:text-slate-700 transition-colors text-xs font-mono font-bold uppercase cursor-pointer"
              >
                Reset Defaults
              </button>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="flex-grow px-3.5 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-mono font-bold uppercase shadow-md shadow-indigo-500/10 hover:shadow-indigo-500/20 transition-all cursor-pointer text-center"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
