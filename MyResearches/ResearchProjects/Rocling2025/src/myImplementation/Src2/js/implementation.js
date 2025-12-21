<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Logit-Based GOP Score Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .loader {
            border-top-color: #3498db;
            -webkit-animation: spinner 1.5s linear infinite;
            animation: spinner 1.5s linear infinite;
        }
        @-webkit-keyframes spinner {
            0% { -webkit-transform: rotate(0deg); }
            100% { -webkit-transform: rotate(360deg); }
        }
        @keyframes spinner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="w-full max-w-4xl bg-white rounded-2xl shadow-lg p-6 md:p-8 space-y-6">
        <!-- Header -->
        <div class="text-center">
            <h1 class="text-2xl md:text-3xl font-bold text-gray-800">Logit-Based GOP Score Analyzer</h1>
            <p class="text-gray-500 mt-2">
                An interactive tool to reproduce the methods from the paper 
                <a href="https://arxiv.org/abs/2506.12067" target="_blank" class="text-blue-500 hover:underline">"Evaluating Logit-Based GOP Scores for Mispronunciation Detection"</a>.
            </p>
        </div>

        <!-- Input Section -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="space-y-4">
                <div>
                    <label for="audio-upload" class="block text-sm font-medium text-gray-700 mb-1">1. Upload Audio File</label>
                    <input type="file" id="audio-upload" accept="audio/*" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 transition-colors duration-200 cursor-pointer">
                </div>
                <div>
                    <label for="transcript" class="block text-sm font-medium text-gray-700 mb-1">2. Enter Transcript (ARPAbet)</label>
                    <textarea id="transcript" rows="4" class="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition" placeholder="Enter space-separated phonemes, e.g., hh ah l ow w er l d"></textarea>
                     <p class="text-xs text-gray-400 mt-1">Note: This is a simplified phoneme alignment. For best results, use a clear recording and matching transcript.</p>
                </div>
            </div>
            <div class="flex flex-col justify-between items-center bg-gray-50 rounded-lg p-4">
                 <div id="status-container" class="text-center space-y-2">
                    <div id="loader" class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-12 w-12 mx-auto hidden"></div>
                    <p id="status-text" class="text-gray-600 font-medium">Loading model, please wait...</p>
                </div>
                <button id="calculate-btn" class="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed" disabled>
                    Calculate GOP Scores
                </button>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results-container" class="hidden">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Analysis Results</h2>
            <div class="overflow-x-auto border border-gray-200 rounded-lg">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Phoneme</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Start (s)</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">End (s)</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">GOP_DNN</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">GOP_MaxLogit</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">GOP_Margin</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">GOP_VarLogit</th>
                            <th class="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">GOP_Combined</th>
                        </tr>
                    </thead>
                    <tbody id="results-body" class="bg-white divide-y divide-gray-200">
                        <!-- Results will be injected here -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script type="module">
        // Import the pipeline function from the Transformers.js library
        import { pipeline, AutoProcessor } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

        // --- DOM Element References ---
        const audioUpload = document.getElementById('audio-upload');
        const transcriptInput = document.getElementById('transcript');
        const calculateBtn = document.getElementById('calculate-btn');
        const statusText = document.getElementById('status-text');
        const loader = document.getElementById('loader');
        const resultsContainer = document.getElementById('results-container');
        const resultsBody = document.getElementById('results-body');

        // --- App State ---
        let model, processor;
        const MODEL_NAME = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft';

        // --- Math Helper Functions ---
        const MathUtils = {
            softmax(arr) {
                const exps = arr.map(x => Math.exp(x));
                const sumExps = exps.reduce((a, b) => a + b, 0);
                return exps.map(e => e / sumExps);
            },
            mean(arr) {
                return arr.reduce((a, b) => a + b, 0) / arr.length;
            },
            variance(arr) {
                const arrMean = this.mean(arr);
                return this.mean(arr.map(x => (x - arrMean) ** 2));
            }
        };

        // --- GOP Calculator Module ---
        const GOPCalculator = {
            alpha: 0.5, // As defined in the paper for GOP_Combined

            calculateGOP_DNN(logitsSlice, targetId) {
                const probabilities = logitsSlice.map(frameLogits => {
                    const softmaxValue = MathUtils.softmax(frameLogits);
                    return softmaxValue[targetId];
                });
                const meanProb = MathUtils.mean(probabilities);
                return -Math.log(meanProb);
            },

            calculateGOP_MaxLogit(logitsSlice, targetId) {
                const targetLogits = logitsSlice.map(frameLogits => frameLogits[targetId]);
                return Math.max(...targetLogits);
            },

            calculateGOP_Margin(logitsSlice, targetId) {
                const margins = logitsSlice.map(frameLogits => {
                    const targetLogit = frameLogits[targetId];
                    const maxCompetitorLogit = Math.max(...frameLogits.filter((_, i) => i !== targetId));
                    return targetLogit - maxCompetitorLogit;
                });
                return MathUtils.mean(margins);
            },

            calculateGOP_VarLogit(logitsSlice, targetId) {
                const targetLogits = logitsSlice.map(frameLogits => frameLogits[targetId]);
                return MathUtils.variance(targetLogits);
            },

            calculateGOP_Combined(gopMargin, gopDnn) {
                return this.alpha * gopMargin - (1 - this.alpha) * gopDnn;
            }
        };

        // --- Core Application Logic ---

        /**
         * Initializes the AI model and processor. Updates UI accordingly.
         */
        async function initializeModel() {
            try {
                statusText.textContent = 'Loading model (~1.2 GB)... This may take a moment.';
                loader.classList.remove('hidden');
                
                model = await pipeline('audio-classification', MODEL_NAME, {
                    quantized: false, // Use full precision for better logit analysis
                });
                processor = await AutoProcessor.from_pretrained(MODEL_NAME);

                statusText.textContent = 'Model loaded. Ready to analyze.';
                loader.classList.add('hidden');
                calculateBtn.disabled = false;
            } catch (error) {
                console.error('Model initialization error:', error);
                statusText.textContent = 'Failed to load model. Please refresh the page.';
                loader.classList.add('hidden');
            }
        }

        /**
         * Reads an audio file, decodes, and resamples it to 16kHz.
         * @param {File} file The audio file to process.
         * @returns {Promise<Float32Array>} The resampled audio data.
         */
        async function processAudioFile(file) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await file.arrayBuffer();
            const decodedData = await audioContext.decodeAudioData(arrayBuffer);
            
            const targetSampleRate = 16000;
            if (decodedData.sampleRate === targetSampleRate) {
                return decodedData.getChannelData(0);
            }

            // Resample audio if necessary
            const offlineContext = new OfflineAudioContext(
                decodedData.numberOfChannels,
                decodedData.duration * targetSampleRate,
                targetSampleRate
            );
            const bufferSource = offlineContext.createBufferSource();
            bufferSource.buffer = decodedData;
            bufferSource.connect(offlineContext.destination);
            bufferSource.start();
            const resampledData = await offlineContext.startRendering();
            return resampledData.getChannelData(0);
        }

        /**
         * Performs simplified phoneme alignment by grouping consecutive argmax predictions.
         * This is an approximation of the CTC-segmentation used in the paper.
         * @param {Float32Array[]} logits The raw logits from the model.
         * @returns {Array} An array of phoneme segments.
         */
        function simplifiedAlignment(logits) {
            const phonemeLabels = Object.values(processor.tokenizer.id_to_token);
            const frameDuration = model.model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate;
            
            let segments = [];
            if (logits.length === 0) return segments;

            let currentPhonemeId = -1;
            let segmentStartFrame = 0;

            for (let i = 0; i < logits.length; i++) {
                const frameLogits = logits[i];
                const predictedId = frameLogits.indexOf(Math.max(...frameLogits));

                if (i === 0) {
                    currentPhonemeId = predictedId;
                }

                if (predictedId !== currentPhonemeId) {
                    // End of a segment
                    const phoneme = phonemeLabels[currentPhonemeId].toLowerCase();
                    if (phoneme !== '<pad>') { // Ignore padding tokens
                        segments.push({
                            phoneme: phoneme,
                            startFrame: segmentStartFrame,
                            endFrame: i - 1,
                            startTime: segmentStartFrame * frameDuration,
                            endTime: i * frameDuration,
                        });
                    }
                    currentPhonemeId = predictedId;
                    segmentStartFrame = i;
                }
            }

            // Add the last segment
            const lastPhoneme = phonemeLabels[currentPhonemeId].toLowerCase();
             if (lastPhoneme !== '<pad>') {
                segments.push({
                    phoneme: lastPhoneme,
                    startFrame: segmentStartFrame,
                    endFrame: logits.length - 1,
                    startTime: segmentStartFrame * frameDuration,
                    endTime: logits.length * frameDuration,
                });
            }
            
            return segments;
        }
        
        /**
         * Matches aligned phonemes with the user's transcript.
         * @param {Array} alignedSegments The segments from simplifiedAlignment.
         * @param {Array<string>} transcriptPhonemes The user-provided phonemes.
         * @returns {Array} Matched segments with their target phoneme ID.
         */
        function matchTranscript(alignedSegments, transcriptPhonemes) {
            // This is a very basic matching algorithm. A more advanced version would use dynamic programming.
            // For this demo, we'll do a simple sequential match.
            const phonemeToId = processor.tokenizer.token_to_id;
            let matched = [];
            let transcriptIndex = 0;

            for (const segment of alignedSegments) {
                if (transcriptIndex < transcriptPhonemes.length) {
                    const targetPhoneme = transcriptPhonemes[transcriptIndex].toLowerCase();
                    // Simple logic: if the predicted phoneme is what we expect, we "consume" the transcript phoneme.
                    if (segment.phoneme === targetPhoneme) {
                        const targetId = phonemeToId[segment.phoneme];
                        if (targetId !== undefined) {
                            matched.push({ ...segment, targetPhoneme, targetId });
                        }
                        transcriptIndex++;
                    } else {
                        // If it doesn't match, we still add the segment but note the mismatch
                        // For GOP, we need the *intended* phoneme's ID. We'll assume the transcript is correct.
                        const targetId = phonemeToId[targetPhoneme];
                        if (targetId !== undefined) {
                             matched.push({ ...segment, phoneme: `[${segment.phoneme}]`, targetPhoneme, targetId });
                        }
                    }
                }
            }
            return matched;
        }


        /**
         * Main handler for the calculate button click event.
         */
        async function handleCalculate() {
            const audioFile = audioUpload.files[0];
            const transcriptText = transcriptInput.value.trim();

            if (!audioFile || !transcriptText) {
                alert('Please provide both an audio file and a transcript.');
                return;
            }

            // Update UI to show processing state
            calculateBtn.disabled = true;
            loader.classList.remove('hidden');
            statusText.textContent = 'Processing audio...';
            resultsContainer.classList.add('hidden');
            resultsBody.innerHTML = '';

            try {
                // 1. Process Audio
                const audioData = await processAudioFile(audioFile);

                // 2. Run Inference
                statusText.textContent = 'Analyzing pronunciation...';
                const output = await model(audioData, { top_k: 0 }); // top_k: 0 returns all logits
                const logits = output.logits[0]; // Get the raw logits tensor

                // 3. Align Phonemes (Simplified)
                const alignedSegments = simplifiedAlignment(logits);

                // 4. Match with transcript
                const transcriptPhonemes = transcriptText.split(/\s+/);
                const matchedSegments = matchTranscript(alignedSegments, transcriptPhonemes);
                
                // 5. Calculate GOP for each segment
                statusText.textContent = 'Calculating GOP scores...';
                const results = matchedSegments.map(segment => {
                    const logitsSlice = logits.slice(segment.startFrame, segment.endFrame + 1);
                    if (logitsSlice.length === 0) return null; // Skip empty segments

                    const gopDnn = GOPCalculator.calculateGOP_DNN(logitsSlice, segment.targetId);
                    const gopMaxLogit = GOPCalculator.calculateGOP_MaxLogit(logitsSlice, segment.targetId);
                    const gopMargin = GOPCalculator.calculateGOP_Margin(logitsSlice, segment.targetId);
                    const gopVarLogit = GOPCalculator.calculateGOP_VarLogit(logitsSlice, segment.targetId);
                    const gopCombined = GOPCalculator.calculateGOP_Combined(gopMargin, gopDnn);

                    return {
                        phoneme: segment.targetPhoneme,
                        startTime: segment.startTime.toFixed(3),
                        endTime: segment.endTime.toFixed(3),
                        gopDnn: gopDnn.toFixed(3),
                        gopMaxLogit: gopMaxLogit.toFixed(3),
                        gopMargin: gopMargin.toFixed(3),
                        gopVarLogit: gopVarLogit.toFixed(3),
                        gopCombined: gopCombined.toFixed(3),
                    };
                }).filter(Boolean); // Remove nulls

                // 6. Render Results
                renderResults(results);
                statusText.textContent = 'Analysis complete.';

            } catch (error) {
                console.error('Calculation error:', error);
                statusText.textContent = `An error occurred: ${error.message}`;
            } finally {
                // Reset UI
                calculateBtn.disabled = false;
                loader.classList.add('hidden');
            }
        }

        /**
         * Renders the calculated results into the results table.
         * @param {Array} results The array of result objects.
         */
        function renderResults(results) {
            if (results.length === 0) {
                resultsBody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-gray-500">No phonemes could be aligned with the transcript.</td></tr>`;
            } else {
                results.forEach(res => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td class="px-4 py-3 whitespace-nowrap"><span class="font-mono bg-gray-100 px-2 py-1 rounded">${res.phoneme}</span></td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.startTime}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.endTime}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.gopDnn}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.gopMaxLogit}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.gopMargin}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.gopVarLogit}</td>
                        <td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">${res.gopCombined}</td>
                    `;
                    resultsBody.appendChild(row);
                });
            }
            resultsContainer.classList.remove('hidden');
        }

        // --- Event Listeners ---
        calculateBtn.addEventListener('click', handleCalculate);

        // --- Initial Load ---
        initializeModel();

    </script>
</body>
</html>
