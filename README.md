# Cartoonizer-Cv

A web-based **Image Cartoonization** application that transforms photos into cartoon-style images using computer vision techniques.

Live Demo: [cartoonizer-cv.vercel.app](https://cartoonizer-cv.vercel.app)

## Features
- **Bilateral Filtering**: Smooths images while preserving edges.
- **Sobel Edge Detection**: Extracts sharp edges to outline features.
- **K-Means++ Color Quantization**: Reduces color palette for a cartoon aesthetic.
- **Interactive Controls**: Adjust color levels, edge intensity, and smoothness in real-time.
- **Download Results**: Save your cartoonized image with a single click.

## Technologies
- **HTML5** & **CSS3** for responsive UI
- **JavaScript** (ES6+) for image processing logic
- **Canvas API** for pixel-level manipulation

## Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/JoelJoshi2002/Cartoonizer-Cv.git
   cd Cartoonizer-Cv
   ```
2. **Open `index.html`** in your favorite web browser.
3. **Upload an image** (JPG, PNG, GIF, BMP).
4. **Choose an algorithm** and tweak parameters:
   - **Color Levels**: Number of colors in output.
   - **Edge Intensity**: Strength of detected outlines.
   - **Smoothness**: Degree of smoothing.
5. **Click "Apply Cartoonization"** to generate your cartoon image.
6. **Download** the result by clicking the download icon on the output.

## Project Structure
```
├── index.html       # Main UI
├── style.css        # Styling and layout
└── script.js        # Image processing algorithms
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve algorithms, add new features, or enhance the UI.

