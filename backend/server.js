/*
* Deepfake Project Backend Server
* Connects the React frontend to the Python AI scripts.
*/

const express = require('express');
const multer = require('multer');
const { PythonShell } = require('python-shell');
const cors = require('cors');
const path = require('path');
const fs = require('fs'); // Node.js File System module

const app = express();
const PORT = 5000;

// --- Middleware ---
app.use(cors());
app.use(express.json());

// --- Static File Serving ---
const outputDir = path.join(__dirname, 'outputs');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}
app.use('/outputs', express.static(outputDir));

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// --- THIS IS THE FIX ---
// We configure multer to use diskStorage to have more control
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir); // Save files to the 'uploads/' directory
  },
  filename: function (req, file, cb) {
    // Create a unique filename but *keep* the original file extension
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

// Use our new storage configuration
const upload = multer({ storage: storage });
// -----------------------


// --- API Routes ---

/**
 * GENERATE ROUTE
 */
app.post('/generate', upload.fields([{ name: 'source' }, { name: 'target' }]), (req, res) => {
    console.log("Received /generate request");

    // The paths will now have the correct file extensions
    const sourcePath = req.files.source[0].path;
    const targetPath = req.files.target[0].path;
    
    const outputFileName = `output_${Date.now()}.mp4`;
    const outputPath = path.join(outputDir, outputFileName);

    let options = {
      mode: 'text',
      pythonOptions: ['-u'],
      
      // *** IMPORTANT: This is set to your *generation* venv
      pythonPath: 'E:\\DeepFakeProject\\generation\\venv_generation\\Scripts\\python.exe', 
      
      scriptPath: path.resolve(__dirname, '../generation/roop/'), 
      
      args: [
        '-s', sourcePath, 
        '-t', targetPath, 
        '-o', outputPath, 
        
        // NOTE: This is set to 'cpu' because your local GPU setup failed.
        // This will be SLOW, but it will WORK.
        // If you ever fix your local GPU, you can change 'cpu' to 'cuda'
        '--execution-provider', 'cpu' 
      ]
    };

    console.log("Running Roop script with options:", options.args);

    PythonShell.run('run.py', options).then(results => {
      console.log('Roop script finished.');
      console.log('Results (from Python):', results);
      
      // Check if the script itself reported an error
      if (results && results.some(line => line.includes('error'))) {
          throw new Error(results.join('\n'));
      }

      res.send({ outputUrl: `/outputs/${outputFileName}` }); 

    }).catch(err => {
      console.error("Error running Roop script:", err);
      res.status(500).send({ error: 'Failed to generate deepfake', details: err.message || err });
    });
});

/**
 * DETECT ROUTE
 */
app.post('/detect', upload.single('video'), (req, res) => {
    console.log("Received /detect request");

    const videoPath = req.file.path;

    let options = {
      mode: 'text',
      pythonOptions: ['-u'], 
      
      // *** IMPORTANT: This is set to your *detection* venv
      pythonPath: 'E:\\DeepFakeProject\\detection\\venv_detection\\Scripts\\python.exe',
      
      scriptPath: path.resolve(__dirname, '../detection/'), 
      
      args: [
          videoPath, 
          // We pass the path to the trained model
          path.resolve(__dirname, '../detection/deepfake_model.h5')
        ]
    };

    console.log("Running Detection script with options:", options.args);
    
    PythonShell.run('detect.py', options).then(results => {
      console.log('Detection script finished.');
      console.log('Results (from Python):', results);
      
      const resultJson = results ? results[results.length - 1] : null; 
      
      if (!resultJson) {
           throw new Error("Python script did not return any output.");
      }

      try {
        const result = JSON.parse(resultJson); 
        if (result.error) {
            console.error("Python script error:", result.error);
            res.status(500).send({ error: 'Detection script failed', details: result.error });
        } else {
            res.send(result); 
        }
      } catch (parseErr) {
        console.error("Error parsing detection result:", parseErr, "Raw result was:", resultJson);
        res.status(500).send({ error: 'Failed to parse detection result', details: parseErr.message });
      }

    }).catch(err => {
      console.error("Error running detection script:", err);
      res.status(500).send({ error: 'Failed to detect deepfake', details: err.message || err });
    });
});

// Start the server
app.listen(PORT, () => console.log(`Backend server running on http://localhost:${PORT}`));

