/**
 * LoanLens — Client-Side Whisper Web Worker
 * 
 * Runs open-source Whisper (Xenova/whisper-tiny) entirely inside the user's browser
 * via WebAssembly (WASM). No API key required, zero external cloud cost, 100% private.
 */

// Global pipeline singleton instance inside the worker
let transcriber = null;
let isLoading = false;

// Initialize Transformers.js environment
async function initTransformers() {
  if (self.transformers) {
    return self.transformers;
  }
  try {
    importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
    if (self.transformers) {
      self.transformers.env.allowLocalModels = false;
      self.transformers.env.useBrowserCache = true;
      return self.transformers;
    }
  } catch (err) {
    console.warn('Could not load transformers via importScripts, attempting ES import:', err);
  }

  try {
    const module = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    module.env.allowLocalModels = false;
    module.env.useBrowserCache = true;
    return module;
  } catch (err) {
    throw new Error('Failed to load Transformers.js runtime: ' + err.message);
  }
}

// Load Whisper Pipeline
async function getTranscriber(progressCallback) {
  if (transcriber) return transcriber;
  if (isLoading) {
    // Wait until pipeline finishes loading
    while (isLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    return transcriber;
  }

  isLoading = true;
  try {
    const tf = await initTransformers();
    const pipeline = tf.pipeline;

    transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny', {
      quantized: true,
      progress_callback: (data) => {
        if (typeof progressCallback === 'function') {
          progressCallback(data);
        }
      }
    });

    isLoading = false;
    return transcriber;
  } catch (err) {
    isLoading = false;
    throw err;
  }
}

// Worker message listener
self.onmessage = async (e) => {
  const { action, id, audio, language, modelName } = e.data;

  if (action === 'load') {
    try {
      self.postMessage({ id, status: 'loading', message: 'Loading on-device Whisper model...' });
      await getTranscriber((progress) => {
        self.postMessage({ id, status: 'progress', progress });
      });
      self.postMessage({ id, status: 'ready', message: 'Whisper model ready for local inference' });
    } catch (error) {
      self.postMessage({ id, status: 'error', error: error.message || 'Failed to load Whisper model' });
    }
    return;
  }

  if (action === 'transcribe') {
    try {
      if (!audio || !(audio instanceof Float32Array)) {
        throw new Error('Invalid audio data provided. Expected Float32Array (16kHz mono).');
      }

      // Ensure model is ready
      const pipe = await getTranscriber((progress) => {
        self.postMessage({ id, status: 'progress', progress });
      });

      self.postMessage({ id, status: 'transcribing', message: 'Transcribing audio locally on your device...' });

      const langCode = (language || 'en').split('-')[0].toLowerCase();
      const options = {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true
      };

      if (langCode && langCode !== 'auto') {
        options.language = langCode;
      }

      const result = await pipe(audio, options);

      self.postMessage({
        id,
        status: 'complete',
        result: {
          text: result.text ? result.text.trim() : '',
          chunks: result.chunks || []
        }
      });
    } catch (error) {
      self.postMessage({ id, status: 'error', error: error.message || 'Local transcription failed' });
    }
    return;
  }
};
