class AdvancedCartoonizer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.originalImageData = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        const imageInput = document.getElementById('imageInput');
        const processBtn = document.getElementById('processBtn');
        const colorSlider = document.getElementById('colorReduction');
        const edgeSlider = document.getElementById('edgeIntensity');
        const smoothSlider = document.getElementById('smoothness');

        imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        processBtn.addEventListener('click', () => this.processImage());

        // Update slider values display
        colorSlider.addEventListener('input', (e) => {
            document.getElementById('colorValue').textContent = e.target.value + ' colors';
        });

        edgeSlider.addEventListener('input', (e) => {
            document.getElementById('edgeValue').textContent = e.target.value;
        });

        smoothSlider.addEventListener('input', (e) => {
            document.getElementById('smoothValue').textContent = e.target.value;
        });
    }

    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Clear previous results
                document.getElementById('results').innerHTML = '';
                this.loadImage(img);
                document.getElementById('processBtn').disabled = false;
                this.updateStatus('Image loaded successfully! Ready to cartoonize.');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    loadImage(img) {
        // Create canvas for processing
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');

        // Resize image if too large for better performance
        const maxSize = 800;
        let { width, height } = img;
        
        if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width *= scale;
            height *= scale;
        }

        this.canvas.width = width;
        this.canvas.height = height;
        this.ctx.drawImage(img, 0, 0, width, height);
        
        // Store original image data
        this.originalImageData = this.ctx.getImageData(0, 0, width, height);
        
        // Display original image
        this.displayResult('Original Image', this.canvas);
    }

    async processImage() {
        if (!this.originalImageData) return;

        const algorithm = document.getElementById('algorithm').value;
        const colorLevels = parseInt(document.getElementById('colorReduction').value);
        const edgeIntensity = parseInt(document.getElementById('edgeIntensity').value);
        const smoothness = parseInt(document.getElementById('smoothness').value);

        this.showProgress();
        this.updateStatus('Processing image...');

        // Clear previous results except original
        const results = document.getElementById('results');
        const children = Array.from(results.children);
        children.slice(1).forEach(child => child.remove());

        try {
            await this.sleep(100); // Allow UI to update

            let processedResults = {};

            switch (algorithm) {
                case 'hybrid':
                    processedResults = await this.hybridCartoonization(colorLevels, edgeIntensity, smoothness);
                    break;
                case 'bilateral':
                    processedResults = await this.bilateralFocusMethod(smoothness);
                    break;
                case 'kmeans':
                    processedResults = await this.kMeansFocusMethod(colorLevels);
                    break;
                case 'edge':
                processedResults = await this.edgeEnhancementMethod(edgeIntensity);
                break
            }

            // Display all results
            Object.entries(processedResults).forEach(([name, imageData]) => {
                if (imageData) {
                    const canvas = this.createCanvasFromImageData(imageData);
                    this.displayResult(name, canvas);
                }
            });

            this.hideProgress();
            this.updateStatus('✅ Cartoonization complete! Scroll down to see results.');

        } catch (error) {
            this.hideProgress();
            this.updateStatus('❌ Error processing image: ' + error.message);
            console.error('Processing error:', error);
        }
    }

    // ADVANCED COMPUTER VISION ALGORITHMS IMPLEMENTATION
    async hybridCartoonization(colorLevels, edgeIntensity, smoothness) {
        edgeIntensity = Math.abs(edgeIntensity * 10 - 200);
        this.updateProgress(20);

        const bilateralFiltered = this.applyBilateralFilter(this.originalImageData, smoothness);
        this.updateProgress(40);
        await this.sleep(50);

        const quantized = this.applyKMeansQuantization(bilateralFiltered, colorLevels);
        this.updateProgress(60);
        await this.sleep(50);

        const edges = this.sobelEdgeDetection(this.originalImageData, edgeIntensity);
        this.updateProgress(80);
        await this.sleep(50);

        const final = this.sophisticatedBlending(quantized, edges);
        this.updateProgress(100);

        return {
            'Bilateral Filtered': bilateralFiltered,
            'Color Quantized': quantized,
            'Edge Map': edges,
            'Final Cartoon': final
        };
    }

sobelEdgeDetection(imageData, edgeThreshold = 50) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;

    // Validate input
    if (!data || data.length !== width * height * 4) {
        throw new Error('Invalid image data: data array length does not match dimensions');
    }

    // Convert to grayscale
    const gray = new Uint8ClampedArray(width * height);
    for (let i = 0; i < data.length; i += 4) {
        gray[i / 4] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    // Sobel kernels
    const sobelX = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ];
    const sobelY = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ];

    const result = new Uint8ClampedArray(width * height * 4);

    // Initialize result array with white pixels (255) and full opacity
    for (let i = 0; i < result.length; i += 4) {
        result[i] = result[i + 1] = result[i + 2] = 255;
        result[i + 3] = 255;
    }

    // Apply Sobel operator
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            // Convolve with Sobel kernels
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const nx = x + dx;
                    const ny = y + dy;

                    // Ensure we don't access out-of-bounds
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const pixelIndex = (ny * width + nx);
                        const pixelValue = gray[pixelIndex] || 0; // Fallback to 0 if undefined
                        sumX += pixelValue * sobelX[dy + 1][dx + 1];
                        sumY += pixelValue * sobelY[dy + 1][dx + 1];
                    }
                }
            }

            // Calculate gradient magnitude
            const magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
            const edge = magnitude > edgeThreshold ? 0 : 255;
            const idx = (y * width + x) * 4;
            result[idx] = result[idx + 1] = result[idx + 2] = edge;
            result[idx + 3] = 255;
        }
    }

    return new ImageData(result, width, height);
}
    async edgeEnhancementMethod(edgeIntensity) {
    console.log('edgeEnhancementMethod called with edgeIntensity:', edgeIntensity, 'originalImageData:', {
        width: this.originalImageData?.width,
        height: this.originalImageData?.height,
        dataLength: this.originalImageData?.data?.length
    });
    edgeIntensity = Math.abs(edgeIntensity * 10 - 140);

    const edges1 = this.sobelEdgeDetection(this.originalImageData, edgeIntensity);
    await this.sleep(100);
    const edges2 = this.sobelEdgeDetection(this.originalImageData, edgeIntensity + 5);
    
    return {
        'Standard Edge Detection': edges1,
        'Enhanced Edge Detection': edges2
    };
}

    async bilateralFocusMethod(smoothness) {
        const filtered1 = this.applyBilateralFilter(this.originalImageData, smoothness);
        await this.sleep(100);
        const filtered2 = this.applyBilateralFilter(filtered1, smoothness * 0.7);
        
        return {
            'Single Pass Bilateral': filtered1,
            'Double Pass Bilateral': filtered2
        };
    }

    async kMeansFocusMethod(colorLevels) {
        const quantized4 = this.applyKMeansQuantization(this.originalImageData, Math.max(4, colorLevels - 2));
        await this.sleep(100);
        const quantized8 = this.applyKMeansQuantization(this.originalImageData, colorLevels);
        await this.sleep(100);
        const quantized12 = this.applyKMeansQuantization(this.originalImageData, colorLevels + 2);
        
        return {
            [`${Math.max(4, colorLevels - 2)} Colors`]: quantized4,
            [`${colorLevels} Colors`]: quantized8,
            [`${colorLevels + 2} Colors`]: quantized12
        };
    }
    // CORE COMPUTER VISION ALGORITHMS

    applyBilateralFilter(imageData, intensity) {
        // Bilateral filter implementation - preserves edges while smoothing
        const data = new Uint8ClampedArray(imageData.data);
        const width = imageData.width;
        const height = imageData.height;
        
        const sigmaColor = intensity * 5;
        const sigmaSpace = intensity * 2;
        const kernelSize = 5;
        const half = Math.floor(kernelSize / 2);

        for (let y = half; y < height - half; y++) {
            for (let x = half; x < width - half; x++) {
                let totalWeight = 0;
                let sumR = 0, sumG = 0, sumB = 0;
                
                const centerIdx = (y * width + x) * 4;
                const centerR = data[centerIdx];
                const centerG = data[centerIdx + 1];
                const centerB = data[centerIdx + 2];

                for (let dy = -half; dy <= half; dy++) {
                    for (let dx = -half; dx <= half; dx++) {
                        const ny = y + dy;
                        const nx = x + dx;
                        const idx = (ny * width + nx) * 4;
                        
                        const r = data[idx];
                        const g = data[idx + 1];
                        const b = data[idx + 2];

                        // Spatial weight
                        const spatialDist = dx * dx + dy * dy;
                        const spatialWeight = Math.exp(-spatialDist / (2 * sigmaSpace * sigmaSpace));

                        // Color weight
                        const colorDist = Math.pow(r - centerR, 2) + Math.pow(g - centerG, 2) + Math.pow(b - centerB, 2);
                        const colorWeight = Math.exp(-colorDist / (2 * sigmaColor * sigmaColor));

                        const weight = spatialWeight * colorWeight;
                        totalWeight += weight;

                        sumR += r * weight;
                        sumG += g * weight;
                        sumB += b * weight;
                    }
                }

                if (totalWeight > 0) {
                    data[centerIdx] = sumR / totalWeight;
                    data[centerIdx + 1] = sumG / totalWeight;
                    data[centerIdx + 2] = sumB / totalWeight;
                }
            }
        }

        return new ImageData(data, width, height);
    }

    applyKMeansQuantization(imageData, k) {
        // Advanced K-means++ color quantization
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        
        // Extract pixels for clustering
        const pixels = [];
        for (let i = 0; i < data.length; i += 4) {
            pixels.push([data[i], data[i + 1], data[i + 2]]);
        }

        // K-means++ initialization
        const centroids = this.kMeansPlusPlus(pixels, k);
        
        // Perform K-means clustering
        const finalCentroids = this.performKMeans(pixels, centroids, 10);

        // Assign each pixel to nearest centroid
        const quantizedData = new Uint8ClampedArray(data);
        for (let i = 0; i < pixels.length; i++) {
            const pixel = pixels[i];
            let minDist = Infinity;
            let closestCentroid = finalCentroids[0];

            for (const centroid of finalCentroids) {
                const dist = this.euclideanDistance(pixel, centroid);
                if (dist < minDist) {
                    minDist = dist;
                    closestCentroid = centroid;
                }
            }

            const idx = i * 4;
            quantizedData[idx] = Math.round(closestCentroid[0]);
            quantizedData[idx + 1] = Math.round(closestCentroid[1]);
            quantizedData[idx + 2] = Math.round(closestCentroid[2]);
        }

        return new ImageData(quantizedData, width, height);
    }

    kMeansPlusPlus(points, k) {
        // K-means++ initialization for better clustering
        const centroids = [];
        
        // Choose first centroid randomly
        centroids.push(points[Math.floor(Math.random() * points.length)]);

        // Choose remaining centroids
        for (let i = 1; i < k; i++) {
            const distances = points.map(point => {
                let minDist = Infinity;
                for (const centroid of centroids) {
                    const dist = this.euclideanDistance(point, centroid);
                    minDist = Math.min(minDist, dist);
                }
                return minDist * minDist;
            });

            const totalDist = distances.reduce((sum, d) => sum + d, 0);
            const threshold = Math.random() * totalDist;
            
            let cumulative = 0;
            for (let j = 0; j < points.length; j++) {
                cumulative += distances[j];
                if (cumulative >= threshold) {
                    centroids.push(points[j]);
                    break;
                }
            }
        }

        return centroids;
    }

    performKMeans(points, centroids, maxIterations) {
        let currentCentroids = centroids.map(c => [...c]);

        for (let iter = 0; iter < maxIterations; iter++) {
            const clusters = Array.from({ length: centroids.length }, () => []);

            // Assign points to closest centroids
            for (const point of points) {
                let minDist = Infinity;
                let clusterIndex = 0;

                for (let i = 0; i < currentCentroids.length; i++) {
                    const dist = this.euclideanDistance(point, currentCentroids[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIndex = i;
                    }
                }

                clusters[clusterIndex].push(point);
            }

            // Update centroids
            const newCentroids = clusters.map(cluster => {
                if (cluster.length === 0) return currentCentroids[clusters.indexOf(cluster)];
                
                const sum = cluster.reduce((acc, point) => [
                    acc[0] + point[0],
                    acc[1] + point[1],
                    acc[2] + point[2]
                ], [0, 0, 0]);

                return [
                    sum[0] / cluster.length,
                    sum[1] / cluster.length,
                    sum[2] / cluster.length
                ];
            });

            // Check for convergence
            let converged = true;
            for (let i = 0; i < currentCentroids.length; i++) {
                if (this.euclideanDistance(currentCentroids[i], newCentroids[i]) > 1) {
                    converged = false;
                    break;
                }
            }

            currentCentroids = newCentroids;
            if (converged) break;
        }

        return currentCentroids;
    }

    sophisticatedBlending(quantized, edges) {
        // Combine color-quantized image with edges using sophisticated blending
        const width = quantized.width;
        const height = quantized.height;
        const result = new Uint8ClampedArray(width * height * 4);
        const quantizedData = quantized.data;
        const edgesData = edges.data;

        for (let i = 0; i < quantizedData.length; i += 4) {
            // If edge pixel is black, use edge color (black)
            if (edgesData[i] === 0) {
                result[i] = 0;     // R
                result[i + 1] = 0; // G
                result[i + 2] = 0; // B
            } else {
                // Otherwise use quantized color
                result[i] = quantizedData[i];
                result[i + 1] = quantizedData[i + 1];
                result[i + 2] = quantizedData[i + 2];
            }
            result[i + 3] = 255; // Alpha
        }

        return new ImageData(result, width, height);
    }

    // UTILITY METHODS

    euclideanDistance(a, b) {
        return Math.sqrt(
            Math.pow(a[0] - b[0], 2) +
            Math.pow(a[1] - b[1], 2) +
            Math.pow(a[2] - b[2], 2)
        );
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    createCanvasFromImageData(imageData) {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        return canvas;
    }

    displayResult(title, canvas) {
        const results = document.getElementById('results');
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const titleElement = document.createElement('h3');
        titleElement.textContent = title;
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'result-image-container';
        
        // Create download button
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'download-btn';
        downloadBtn.title = 'Download this image';
        downloadBtn.innerHTML = `
            <svg viewBox="0 0 24 24">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
        `;
        
        // Add click event to download the image
        downloadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.downloadCanvas(canvas, title);
        });
        
        imageContainer.appendChild(canvas);
        imageContainer.appendChild(downloadBtn);
        
        resultItem.appendChild(titleElement);
        resultItem.appendChild(imageContainer);
        results.appendChild(resultItem);
    }

    downloadCanvas(canvas, title) {
        // Create a temporary link to download the canvas as an image
        const link = document.createElement('a');
        link.download = `cartoonized_${title.toLowerCase().replace(/\s+/g, '_')}.png`;
        link.href = canvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    showProgress() {
        document.querySelector('.progress').style.display = 'block';
        document.querySelector('.progress-bar').style.width = '0%';
    }

    hideProgress() {
        document.querySelector('.progress').style.display = 'none';
    }

    updateProgress(percent) {
        document.querySelector('.progress-bar').style.width = percent + '%';
    }

    updateStatus(message) {
        document.querySelector('.status').textContent = message;
    }
}

// Initialize the cartoonizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AdvancedCartoonizer();
});
