/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Interactive 3D Knowledge Graph Canvas Component.
 * Implements perspective 3D-to-2D projection, dragging, auto-rotation, depth-cueing, and flow particles.
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Triple } from '../utils/kge';
import { Maximize2, Minimize2, Eye, EyeOff, RotateCcw } from 'lucide-react';

interface Graph3DProps {
  entityEmbeddings: { [entity: string]: number[] }; // 3D projected coordinates
  triples: Triple[];
  hoveredNode: string | null;
  setHoveredNode: (node: string | null) => void;
  selectedNode: string | null;
  setSelectedNode: (node: string | null) => void;
  syntheticNames?: string[];
  nodeColors?: { [entity: string]: { start: string; end: string } };
  projectionMode?: 'pca' | 'umap';
  setProjectionMode?: (mode: 'pca' | 'umap') => void;
  clusteringMode?: 'default' | 'kmeans';
  setClusteringMode?: (mode: 'default' | 'kmeans') => void;
  numClusters?: number;
  setNumClusters?: (num: number) => void;
}

interface ProjectedNode {
  id: string;
  name: string;
  x: number;
  y: number;
  z: number;
  projX: number;
  projY: number;
  projZ: number;
  radius: number;
}

export const Graph3D: React.FC<Graph3DProps> = ({
  entityEmbeddings,
  triples,
  hoveredNode,
  setHoveredNode,
  selectedNode,
  setSelectedNode,
  syntheticNames = [],
  nodeColors,
  projectionMode = 'pca',
  setProjectionMode,
  clusteringMode = 'default',
  setClusteringMode,
  numClusters = 5,
  setNumClusters,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Interaction State
  const [renderMode, setRenderMode] = useState<'3d' | '2d'>('2d');
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const yawRef = useRef<number>(0.5); // Y-rotation
  const pitchRef = useRef<number>(0.3); // X-rotation
  const [scale, setScale] = useState<number>(200); // Zoom scale
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [autoRotate, setAutoRotate] = useState<boolean>(true);

  // Dimensions
  const [dimensions, setDimensions] = useState({ width: 600, height: 450 });

  // Custom colors for relations (strictly cohesive Immersive theme: cyan, blue, indigo, violet, teal)
  const relationColors = useMemo(() => {
    const colors = [
      '#6366f1', // indigo
      '#0d9488', // teal
      '#3b82f6', // blue
      '#8b5cf6', // violet
      '#ec4899', // pink
      '#f59e0b', // amber
      '#10b981', // emerald
      '#06b6d4', // cyan
    ];
    const map: { [rel: string]: string } = {};
    let colorIdx = 0;
    triples.forEach(t => {
      if (!map[t.rel]) {
        map[t.rel] = colors[colorIdx % colors.length];
        colorIdx++;
      }
    });
    return map;
  }, [triples]);

  // Track container sizing dynamically
  useEffect(() => {
    if (!containerRef.current) return;
    
    const observer = new ResizeObserver((entries) => {
      if (entries.length === 0) return;
      const { width, height } = entries[0].contentRect;
      setDimensions({
        width: Math.max(width, 300),
        height: Math.max(height, 300),
      });
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Main 3D Rendering loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const centerVec = { x: 0, y: 0, z: 0 };
    const keys = Object.keys(entityEmbeddings);
    
    // Scale embeddings dynamically to fit nice on screen
    let maxDist = 1;
    keys.forEach(k => {
      const vec = entityEmbeddings[k];
      const d = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
      if (d > maxDist) maxDist = d;
    });

    const scaleFactor = 100 / (maxDist || 1);

    const render = () => {
      // Background and reset
      ctx.clearRect(0, 0, dimensions.width, dimensions.height);
      
      // Soft, easy-on-the-eyes light radial gradient (white center fading to slate-50)
      const spaceGrad = ctx.createRadialGradient(
        dimensions.width / 2, dimensions.height / 2, 20,
        dimensions.width / 2, dimensions.height / 2, dimensions.width
      );
      spaceGrad.addColorStop(0, '#ffffff');
      spaceGrad.addColorStop(1, '#f8fafc');
      ctx.fillStyle = spaceGrad;
      ctx.fillRect(0, 0, dimensions.width, dimensions.height);

      // Subtle light grid matching a beautiful, clean paper notebook look
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.03)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < dimensions.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, dimensions.height);
        ctx.stroke();
      }
      for (let y = 0; y < dimensions.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(dimensions.width, y);
        ctx.stroke();
      }

      // Auto-rotation when idle (only for 3D)
      if (renderMode === '3d' && autoRotate && !isDragging) {
        yawRef.current += 0.0015;
      }
      const currentYaw = yawRef.current;

      // Camera Distance
      const cameraDist = 450;
      const cosY = Math.cos(currentYaw);
      const sinY = Math.sin(currentYaw);
      const cosP = Math.cos(pitchRef.current);
      const sinP = Math.sin(pitchRef.current);

      // Perspective projection helper
      const project = (x: number, y: number, z: number) => {
        // Apply scaleFactor to normalize embeddings
        const nx = x * scaleFactor;
        const ny = y * scaleFactor;
        const nz = z * scaleFactor;

        // Yaw (Y-axis rotation)
        const rotX = nx * cosY - nz * sinY;
        const rotZ = nx * sinY + nz * cosY;

        // Pitch (X-axis rotation)
        const rotY = ny * cosP - rotZ * sinP;
        const projZ = ny * sinP + rotZ * cosP;

        // Perspective projection formula
        const factor = scale / (projZ + cameraDist);
        const px = dimensions.width / 2 + rotX * factor;
        const py = dimensions.height / 2 + rotY * factor;

        return { px, py, pz: projZ };
      };

      // 1. Project all nodes
      // Calculate max absolute values for 2D fitting
      let maxAbsX = 1e-5;
      let maxAbsY = 1e-5;
      keys.forEach(k => {
        const vec = entityEmbeddings[k] || [0, 0, 0];
        const absX = Math.abs(vec[0]);
        const absY = Math.abs(vec[1]);
        if (absX > maxAbsX) maxAbsX = absX;
        if (absY > maxAbsY) maxAbsY = absY;
      });

      const fitScaleX = (dimensions.width / 2) * 0.8 / maxAbsX;
      const fitScaleY = (dimensions.height / 2) * 0.8 / maxAbsY;
      const autoFit2DScale = Math.min(fitScaleX, fitScaleY);

      const projectedNodes: ProjectedNode[] = keys.map(k => {
        const vec = entityEmbeddings[k] || [0, 0, 0];
        let px = 0;
        let py = 0;
        let pz = 0;

        if (renderMode === '3d') {
          const projected = project(vec[0], vec[1], vec[2]);
          px = projected.px;
          py = projected.py;
          pz = projected.pz;
        } else { // '2d'
          // Flatten 2D Map perfectly centered & auto-scaled to container bounds, plus the panOffset!
          px = dimensions.width / 2 + vec[0] * autoFit2DScale * (scale / 200) + panOffset.x;
          py = dimensions.height / 2 + vec[1] * autoFit2DScale * (scale / 200) + panOffset.y;
          pz = 0;
        }
        
        let nodeSize = 6;
        if (k === selectedNode) nodeSize = 12;
        else if (k === hoveredNode) nodeSize = 10;
        
        // Depth cueing scaling is only for 3D mode (safeguard against division by zero/negative)
        const denom = pz + cameraDist;
        const safeDenom = denom <= 10 ? 10 : denom;
        const radius = renderMode === '3d' 
          ? nodeSize * (cameraDist / safeDenom) 
          : nodeSize * 1.1;
        
        return {
          id: k,
          name: k,
          x: vec[0],
          y: vec[1],
          z: vec[2],
          projX: isFinite(px) ? px : dimensions.width / 2,
          projY: isFinite(py) ? py : dimensions.height / 2,
          projZ: isFinite(pz) ? pz : 0,
          radius: isFinite(radius) && radius > 0 ? radius : nodeSize,
        };
      });

      // Map node name to projected model for quick link calculation
      const nodeMap = new Map<string, ProjectedNode>();
      projectedNodes.forEach(pn => nodeMap.set(pn.id, pn));

      // 2. Draw Edges / Relations
      triples.forEach(triple => {
        const subNode = nodeMap.get(triple.sub);
        const objNode = nodeMap.get(triple.obj);

        if (!subNode || !objNode) return;

        const isHighlighted = 
          hoveredNode === triple.sub || 
          hoveredNode === triple.obj ||
          selectedNode === triple.sub ||
          selectedNode === triple.obj;

        const isDimmed = (hoveredNode || selectedNode) && !isHighlighted;

        ctx.beginPath();
        ctx.moveTo(subNode.projX, subNode.projY);
        
        // Draw elegant Bezier curves for relationships to prevent line stacking
        const midX = (subNode.projX + objNode.projX) / 2;
        const midY = (subNode.projY + objNode.projY) / 2;
        // Subtle offset orthogonal to vector
        const dx = objNode.projX - subNode.projX;
        const dy = objNode.projY - subNode.projY;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const offsetAmount = 15;
        const cx = midX - (dy / len) * offsetAmount;
        const cy = midY + (dx / len) * offsetAmount;

        ctx.quadraticCurveTo(cx, cy, objNode.projX, objNode.projY);
        
        // Color matching
        const relColor = relationColors[triple.rel] || '#06b6d4';
        ctx.strokeStyle = relColor;
        
        // Depth cueing for lines: further away lines are thinner/more transparent (3D only)
        const avgDepth = (subNode.projZ + objNode.projZ) / 2;
        const depthAlpha = renderMode === '3d' ? Math.max(0.1, 1 - (avgDepth + 100) / 400) : 0.8;
        
        ctx.lineWidth = isHighlighted ? 2.5 : 1;
        ctx.strokeStyle = isHighlighted 
          ? relColor 
          : isDimmed 
            ? 'rgba(15, 23, 42, 0.02)' 
            : `${relColor}${Math.floor(depthAlpha * 60).toString(16).padStart(2, '0')}`;
        ctx.stroke();

        // Relation text on hover/select (neatly paired mono typography)
        if (isHighlighted && !isDimmed) {
          ctx.font = '10px monospace';
          ctx.fillStyle = '#475569'; // slate-600
          ctx.fillText(triple.rel, cx, cy);
        }
      });

      // 3. Draw Nodes sorted by depth (Painter's algorithm: draw further nodes first!)
      const sortedNodes = renderMode === '3d' 
        ? [...projectedNodes].sort((a, b) => b.projZ - a.projZ)
        : [...projectedNodes];

      sortedNodes.forEach(node => {
        // Guard against non-finite or invalid projection values
        if (!isFinite(node.projX) || !isFinite(node.projY)) {
          return;
        }

        const isHovered = hoveredNode === node.id;
        const isSelected = selectedNode === node.id;
        const isHighlighted = isHovered || isSelected || (hoveredNode && triples.some(t => (t.sub === hoveredNode && t.obj === node.id) || (t.obj === hoveredNode && t.sub === node.id)));
        const isDimmed = (hoveredNode || selectedNode) && !isHighlighted;

        // Depth cueing alpha (only 3D)
        const depthAlpha = renderMode === '3d' ? Math.max(0.15, 1 - (node.projZ + 100) / 450) : 0.95;

        const r = Math.max(1, isFinite(node.radius) && node.radius > 0 ? node.radius : 2);

        // Core Sphere Gradient
        const radGrad = ctx.createRadialGradient(
          node.projX - r / 3, node.projY - r / 3, r / 10,
          node.projX, node.projY, r
        );

        const isSynthetic = syntheticNames.includes(node.id);

        let colStart = '#6366f1'; // indigo-500
        let colEnd = '#4f46e5';   // indigo-600

        if (nodeColors && nodeColors[node.id]) {
          colStart = nodeColors[node.id].start;
          colEnd = nodeColors[node.id].end;
        }

        if (isSelected) {
          colStart = '#f43f5e'; // rose-500
          colEnd = '#be123c';   // rose-700
        } else if (isHovered) {
          colStart = '#a855f7'; // purple-500
          colEnd = '#6b21a8';   // purple-700
        } else if (isSynthetic) {
          colStart = '#f59e0b'; // amber-500 (glowing gold)
          colEnd = '#d97706';   // amber-600
        } else if (isHighlighted) {
          colStart = '#0d9488'; // teal-600
          colEnd = '#0f766e';   // teal-700
        }

        const colorA = isDimmed ? 'rgba(203, 213, 225, 0.4)' : colStart;
        const colorB = isDimmed ? 'rgba(148, 163, 184, 0.5)' : colEnd;

        radGrad.addColorStop(0, colorA);
        radGrad.addColorStop(1, colorB);

        // Draw node body
        ctx.beginPath();
        ctx.arc(node.projX, node.projY, Math.max(2, node.radius), 0, 2 * Math.PI);
        ctx.fillStyle = radGrad;
        ctx.fill();

        // Glowing outer halo
        if (isHighlighted && !isDimmed) {
          ctx.beginPath();
          ctx.arc(node.projX, node.projY, node.radius + 4, 0, 2 * Math.PI);
          ctx.strokeStyle = isSelected ? 'rgba(244, 63, 94, 0.25)' : 'rgba(99, 102, 241, 0.25)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Draw Text Label with depth fogging
        const showLabel = showLabels && (!isDimmed || isHighlighted);
        if (showLabel) {
          ctx.font = isSelected ? 'bold 11px "Inter", sans-serif' : '10px "Inter", sans-serif';
          
          // Draw subtle background plate behind text for legibility
          const textWidth = ctx.measureText(node.name).width;
          ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
          ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
          ctx.lineWidth = 1;
          ctx.fillRect(node.projX - textWidth / 2 - 4, node.projY - node.radius - 14, textWidth + 8, 14);
          ctx.strokeRect(node.projX - textWidth / 2 - 4, node.projY - node.radius - 14, textWidth + 8, 14);

          ctx.fillStyle = isSelected 
            ? '#be123c' // rose-700
            : isHovered 
              ? '#6b21a8' // purple-700
              : isDimmed 
                ? 'rgba(148, 163, 184, 0.25)' 
                : `rgba(15, 23, 42, ${Math.max(0.4, depthAlpha)})`;
          ctx.textAlign = 'center';
          ctx.fillText(node.name, node.projX, node.projY - node.radius - 4);
        }
      });

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [entityEmbeddings, triples, scale, isDragging, autoRotate, hoveredNode, selectedNode, dimensions, relationColors, renderMode, panOffset, showLabels, nodeColors]);

  // Handlers for Canvas Interactions
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setAutoRotate(false);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // 1. If dragging, handle 3D Rotation (only valid in 3D mode) or 2D Panning
    if (isDragging) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      if (renderMode === '3d') {
        yawRef.current += dx * 0.007;
        pitchRef.current = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, pitchRef.current + dy * 0.007));
      } else { // '2d'
        setPanOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
      }
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    // 2. Otherwise, check for Node Hovering
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const cameraDist = 450;
    const cosY = Math.cos(yawRef.current);
    const sinY = Math.sin(yawRef.current);
    const cosP = Math.cos(pitchRef.current);
    const sinP = Math.sin(pitchRef.current);

    // Compute maxDist and 2D fitting parameters (must match render block)
    let maxDist = 1;
    let maxAbsX = 1e-5;
    let maxAbsY = 1e-5;
    const keys = Object.keys(entityEmbeddings);
    keys.forEach(k => {
      const vec = entityEmbeddings[k];
      const d = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
      if (d > maxDist) maxDist = d;
      
      const absX = Math.abs(vec[0]);
      const absY = Math.abs(vec[1]);
      if (absX > maxAbsX) maxAbsX = absX;
      if (absY > maxAbsY) maxAbsY = absY;
    });
    const scaleFactor = 100 / (maxDist || 1);
    
    const fitScaleX = (dimensions.width / 2) * 0.8 / maxAbsX;
    const fitScaleY = (dimensions.height / 2) * 0.8 / maxAbsY;
    const autoFit2DScale = Math.min(fitScaleX, fitScaleY);

    let foundNode: string | null = null;
    let closestZ = -Infinity;

    keys.forEach(k => {
      const vec = entityEmbeddings[k];
      let px = 0;
      let py = 0;
      let pz = 0;
      let nodeSize = k === selectedNode ? 12 : 6;
      let radius = nodeSize;

      if (renderMode === '3d') {
        const nx = vec[0] * scaleFactor;
        const ny = vec[1] * scaleFactor;
        const nz = vec[2] * scaleFactor;

        const rotX = nx * cosY - nz * sinY;
        const rotZ = nx * sinY + nz * cosY;
        const rotY = ny * cosP - rotZ * sinP;
        const projZ = ny * sinP + rotZ * cosP;

        const factor = scale / (projZ + cameraDist);
        px = dimensions.width / 2 + rotX * factor;
        py = dimensions.height / 2 + rotY * factor;
        pz = projZ;
        radius = nodeSize * (cameraDist / (pz + cameraDist));
      } else { // '2d'
        px = dimensions.width / 2 + vec[0] * autoFit2DScale * (scale / 200) + panOffset.x;
        py = dimensions.height / 2 + vec[1] * autoFit2DScale * (scale / 200) + panOffset.y;
        pz = 0;
        radius = nodeSize * 1.1;
      }

      const dist = Math.sqrt((mx - px) * (mx - px) + (my - py) * (my - py));

      if (dist <= radius + 5) {
        if (pz > closestZ) { // Hover the top-most node
          closestZ = pz;
          foundNode = k;
        }
      }
    });

    if (foundNode !== hoveredNode) {
      setHoveredNode(foundNode);
    }
  };

  const handleMouseUpOrLeave = () => {
    setIsDragging(false);
  };

  const handleCanvasClick = () => {
    // If we click the blank space, deselect the node
    if (!isDragging) {
      setSelectedNode(hoveredNode);
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setScale(prev => Math.max(50, Math.min(600, prev - e.deltaY * 0.2)));
  };

  const resetView = () => {
    yawRef.current = 0.5;
    pitchRef.current = 0.3;
    setScale(200);
    setPanOffset({ x: 0, y: 0 });
    setAutoRotate(true);
    setSelectedNode(null);
    setHoveredNode(null);
  };

  return (
    <div
      id="graph-container-card"
      ref={containerRef}
      className={isExpanded
        ? "fixed inset-0 z-50 w-screen h-screen bg-white flex flex-col p-4 animate-fade-in"
        : "relative w-full h-[380px] md:h-[450px] rounded-2xl bg-white border border-slate-200/80 overflow-hidden shadow-md shadow-slate-100 flex flex-col"
      }
    >
      {/* Top HUD Overlay */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-1 pointer-events-none">
        <span className="text-xs font-mono text-indigo-600 flex items-center gap-1.5">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
          </span>
          {renderMode === '3d' && '3D KNOWLEDGE GRAPH VISUALIZATION'}
          {renderMode === '2d' && '2D KNOWLEDGE GRAPH VISUALIZATION'}
        </span>
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
          {renderMode === '3d' 
            ? 'Rotate: Drag • Zoom: Scroll • Select: Click Node' 
            : 'Pan: Drag • Zoom: Scroll • Select: Click Node'}
        </span>
      </div>

      {/* Secondary Controls overlay (beneath Top-Left HUD) */}
      <div className="absolute top-14 left-4 right-4 md:right-auto z-10 flex flex-wrap gap-2 items-center pointer-events-auto">
        {/* Projection Selector */}
        {setProjectionMode && (
          <div className="flex bg-slate-50/95 p-0.5 rounded-lg border border-slate-200/80 backdrop-blur-md shadow-xs items-center gap-1">
            <span className="text-[8px] font-mono font-bold text-slate-400 px-1.5 uppercase">Projection</span>
            <button
              onClick={() => setProjectionMode('pca')}
              className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
                projectionMode === 'pca'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              PCA
            </button>
            <button
              onClick={() => setProjectionMode('umap')}
              className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
                projectionMode === 'umap'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              UMAP
            </button>
          </div>
        )}

        {/* Clustering Selector */}
        {setClusteringMode && (
          <div className="flex bg-slate-50/95 p-0.5 rounded-lg border border-slate-200/80 backdrop-blur-md shadow-xs items-center gap-1">
            <span className="text-[8px] font-mono font-bold text-slate-400 px-1.5 uppercase">Coloring</span>
            <button
              onClick={() => setClusteringMode('default')}
              className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
                clusteringMode === 'default'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              Role-based
            </button>
            <button
              onClick={() => setClusteringMode('kmeans')}
              className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
                clusteringMode === 'kmeans'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              K-Means Clusters
            </button>
          </div>
        )}

        {/* K-Means Clusters Count Selector */}
        {clusteringMode === 'kmeans' && setNumClusters && numClusters !== undefined && (
          <div className="flex bg-slate-50/95 p-0.5 rounded-lg border border-slate-200/80 backdrop-blur-md shadow-xs items-center gap-1.5 h-[21px] px-1.5">
            <span className="text-[8px] font-mono font-bold text-slate-400 uppercase whitespace-nowrap">Clusters (K={numClusters})</span>
            <input
              type="range"
              min="2"
              max="10"
              value={numClusters}
              onChange={(e) => setNumClusters(parseInt(e.target.value))}
              className="w-14 h-1 cursor-pointer accent-indigo-600 outline-none"
            />
          </div>
        )}
      </div>

      {/* Buttons HUD Overlay */}
      <div className="absolute top-4 right-4 z-10 flex gap-2 items-center">
        {/* Label Toggle Button */}
        <button
          onClick={() => setShowLabels(!showLabels)}
          className={`p-1 px-2 rounded-lg border text-[9px] font-mono flex items-center gap-1 transition-all pointer-events-auto cursor-pointer ${
            showLabels 
              ? 'bg-indigo-50 border-indigo-200 text-indigo-600 hover:bg-indigo-100/80' 
              : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
          }`}
          title={showLabels ? "Hide Node Labels" : "Show Node Labels"}
        >
          {showLabels ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
          <span>{showLabels ? 'LABELS ON' : 'LABELS OFF'}</span>
        </button>

        {/* Render Mode Selector */}
        <div className="flex bg-slate-100/95 p-0.5 rounded-lg border border-slate-200/80 backdrop-blur-md pointer-events-auto shadow-xs">
          <button
            onClick={() => setRenderMode('3d')}
            className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
              renderMode === '3d'
                ? 'bg-white text-indigo-600 shadow-xs'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            3D
          </button>
          <button
            onClick={() => setRenderMode('2d')}
            className={`px-2 py-0.5 rounded text-[9px] font-bold font-mono transition-all cursor-pointer ${
              renderMode === '2d'
                ? 'bg-white text-indigo-600 shadow-xs'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            2D
          </button>
        </div>

        {renderMode === '3d' && (
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-1 px-2 rounded-lg border text-[9px] font-mono flex items-center gap-1 transition-all pointer-events-auto cursor-pointer ${
              autoRotate 
                ? 'bg-indigo-50 border-indigo-200 text-indigo-600 hover:bg-indigo-100/80' 
                : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
            }`}
            title="Toggle Auto-Rotation"
          >
            {autoRotate ? 'SPIN ON' : 'SPIN OFF'}
          </button>
        )}

        <button
          onClick={resetView}
          className="p-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition-all pointer-events-auto flex items-center justify-center cursor-pointer"
          title="Reset View"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>

        {/* Expand / Minimize Toggle Button */}
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-500 hover:bg-slate-100 hover:text-slate-800 transition-all pointer-events-auto flex items-center justify-center cursor-pointer"
          title={isExpanded ? "Minimize Visualization" : "Expand Visualization"}
        >
          {isExpanded ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
      </div>

      {/* Actual Graphics Canvas */}
      <canvas
        id="kge-3d-canvas"
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUpOrLeave}
        onMouseLeave={handleMouseUpOrLeave}
        onClick={handleCanvasClick}
        onWheel={handleWheel}
        className="w-full h-full cursor-grab active:cursor-grabbing block animate-fade-in"
      />

      {/* Bottom Node Selection HUD */}
      {(selectedNode || hoveredNode) && (
        <div className="absolute bottom-4 left-4 right-4 z-10 p-3 rounded-xl bg-white/95 border border-slate-200/80 backdrop-blur-md flex items-center justify-between text-xs font-mono pointer-events-auto shadow-md shadow-slate-100">
          <div className="flex flex-col gap-0.5">
            <span className="text-[9px] text-slate-450 uppercase tracking-wider font-semibold">ACTIVE SELECTION</span>
            <span className="text-sm font-semibold text-indigo-600">{selectedNode || hoveredNode}</span>
          </div>
          {selectedNode && (
            <button
              onClick={() => setSelectedNode(null)}
              className="text-[10px] text-rose-600 hover:text-rose-700 border border-rose-200 rounded px-2.5 py-0.5 transition-all bg-rose-50 cursor-pointer"
            >
              DESELECT
            </button>
          )}
        </div>
      )}
    </div>
  );
};
