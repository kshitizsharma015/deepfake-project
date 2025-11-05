/*
* This is the main React component for your frontend.
* It manages the state for file uploads, loading, and results.
* It also contains the functions to call your backend server.
*/
import React, { useState } from 'react';
import axios from 'axios'; // For making HTTP requests to the backend
import './App.css'; // Import the CSS file for styling

function App() {
  // State variables to hold the selected files
  const [sourceFile, setSourceFile] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  const [detectFile, setDetectFile] = useState(null);
  
  // State to store the result from the backend (a video URL or a detection score)
  const [result, setResult] = useState(null); 
  // State to show a loading message while the AI is working
  const [isLoading, setIsLoading] = useState(false); 
  // State to show any errors that occur
  const [error, setError] = useState(''); 

  // --- Handler for Generation ---
  const handleGenerate = async () => {
    // Check if both files are selected
    if (!sourceFile || !targetFile) {
      setError('Please select both a source face image and a target media file.');
      return;
    }
    setIsLoading(true);
    setError(''); // Clear previous errors
    setResult(null); // Clear previous results

    // Use FormData to send files to the server
    const formData = new FormData(); 
    formData.append('source', sourceFile);
    formData.append('target', targetFile);

    try {
      // Send the POST request to the /generate endpoint on your backend
      const res = await axios.post('http://localhost:5000/generate', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // The backend sends back a URL to the generated video
      // We must add "http://localhost:5000" to the relative URL
      setResult({ type: 'generate', outputUrl: `http://localhost:5000${res.data.outputUrl}` }); 

    } catch (err) {
      console.error('Generation failed:', err);
      // Display a user-friendly error message
      setError(`Generation failed: ${err.response?.data?.details || err.message}`);
    } finally {
      setIsLoading(false); // Stop loading, whether it succeeded or failed
    }
  };

  // --- Handler for Detection ---
  const handleDetect = async () => {
    // Check if a file is selected
    if (!detectFile) {
      setError('Please select a video file to detect.');
      return;
    }
    setIsLoading(true);
    setError(''); // Clear previous errors
    setResult(null); // Clear previous results

    const formData = new FormData();
    formData.append('video', detectFile);

    try {
      // Send the POST request to the /detect endpoint on your backend
      const res = await axios.post('http://localhost:5000/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Check if the Python script itself sent an error
      if (res.data.error) {
         setError(`Detection script error: ${res.data.error}`);
      } else {
         // Update state with the detection result (JSON object)
         setResult({ type: 'detect', ...res.data });
      }
    } catch (err) {
      console.error('Detection failed:', err);
      setError(`Detection failed: ${err.response?.data?.details || err.message}`);
    } finally {
      setIsLoading(false); // Stop loading
    }
  };

  // --- Render the UI (the HTML structure of your site) ---
  return (
    <div className="App">
      <header className="App-header">
        <h1>Deepfake Tool</h1>
      </header>

      {/* Show a loading message while AI is working */}
      {isLoading && <div className="loading">Processing... Please wait. This may take a while.</div>}

      {/* Show an error message if something went wrong */}
      {error && <div className="error">Error: {error}</div>}

      {/* Generation Section */}
      <section className="card">
        <h2>Generate Deepfake</h2>
        <div className="input-group">
          <label htmlFor="source">Source Face Image:</label>
          <input id="source" type="file" accept="image/*" onChange={e => setSourceFile(e.target.files[0])} />
        </div>
        <div className="input-group">
          <label htmlFor="target">Target Media (Video/Image):</label>
          <input id="target" type="file" accept="video/*,image/*" onChange={e => setTargetFile(e.target.files[0])} />
        </div>
        <button onClick={handleGenerate} disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate Deepfake'}
        </button>
      </section>

      {/* Detection Section */}
      <section className="card">
        <h2>Detect Deepfake</h2>
        <div className="input-group">
          <label htmlFor="detect">Video to Detect:</label>
          <input id="detect" type="file" accept="video/*" onChange={e => setDetectFile(e.target.files[0])} />
        </div>
        <button onClick={handleDetect} disabled={isLoading}>
          {isLoading ? 'Detecting...' : 'Detect Deepfake'}
        </button>
      </section>

      {/* Result Display Section (only shows if 'result' is not null) */}
      {result && !isLoading && (
        <section className="card result">
          <h2>Result ✨</h2>
          
          {/* If the result was from Generation */}
          {result.type === 'generate' && result.outputUrl ? (
            <div>
              <p>Generation Complete!</p>
              {/* Display the generated video */}
              <video src={result.outputUrl} controls width="400" />
               <br/>
               <a href={result.outputUrl} download={`deepfake_${Date.now()}.mp4`}>Download Video</a>
            </div>
          
          /* If the result was from Detection */
          ) : result.type === 'detect' && result.fake_probability !== undefined ? (
            <div>
              <p>Detection Complete!</p>
              <p><strong>Fake Probability:</strong> {(result.fake_probability * 100).toFixed(2)}%</p>
              <p><strong>Verdict:</strong> {result.is_fake ? 'Likely FAKE' : 'Likely REAL'}</p>
            </div>
          
          /* If the result is something else */
          ) : (
             <p>Unexpected result format received.</p>
          )}
        </section>
      )}

      <footer className="ethics-note">
        <p><strong>Ethics Note:</strong> This tool is for educational purposes only. Do not misuse generated content. Always label generated media as "FAKE". Use only public domain or consented data.</p>
      </footer>
    </div>
  );
}

export default App;