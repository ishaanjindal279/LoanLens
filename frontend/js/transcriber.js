/**
 * LoanLens — Local Whisper Transcription Engine
 * 
 * Abstraction layer for running browser-based speech-to-text without API keys.
 * Decodes audio in browser using Web Audio API, then passes 16kHz mono PCM to Web Worker.
 */

class LocalWhisperTranscriber {
  constructor() {
    this.worker = null;
    this.isModelLoaded = false;
    this.isTranscribing = false;
    this.currentTaskId = 0;
    this.callbacks = new Map();
    this.initWorker();
  }

  isSupported() {
    return (
      typeof window !== 'undefined' &&
      typeof window.Worker !== 'undefined' &&
      (typeof window.AudioContext !== 'undefined' || typeof window.webkitAudioContext !== 'undefined')
    );
  }

  initWorker() {
    if (!this.isSupported()) return;
    try {
      if (this.worker) {
        this.worker.terminate();
      }
      this.worker = new Worker('js/whisper-worker.js');
      this.worker.onmessage = (e) => this.handleWorkerMessage(e.data);
      this.worker.onerror = (err) => {
        console.error('Whisper worker error:', err);
      };
    } catch (e) {
      console.warn('Could not initialize Whisper worker:', e);
    }
  }

  handleWorkerMessage(data) {
    const { id, status, message, progress, result, error } = data;
    const cb = this.callbacks.get(id);
    if (!cb) return;

    if (status === 'progress') {
      if (cb.onProgress) {
        let percent = 0;
        if (progress) {
          if (typeof progress.progress === 'number') {
            percent = Math.round(progress.progress);
          } else if (progress.loaded && progress.total) {
            percent = Math.round((progress.loaded / progress.total) * 100);
          }
        }
        cb.onProgress({
          status: 'downloading',
          percent: Math.min(100, Math.max(0, percent)),
          file: progress ? (progress.file || '') : '',
          message: progress && progress.file ? `Downloading ${progress.file} (${percent}%)` : `Preparing model... ${percent}%`
        });
      }
      return;
    }

    if (status === 'loading') {
      if (cb.onProgress) {
        cb.onProgress({ status: 'loading', message: message || 'Loading model into memory...' });
      }
      return;
    }

    if (status === 'transcribing') {
      if (cb.onProgress) {
        cb.onProgress({ status: 'transcribing', message: message || 'Transcribing speech locally...' });
      }
      return;
    }

    if (status === 'ready') {
      this.isModelLoaded = true;
      if (cb.onProgress) {
        cb.onProgress({ status: 'ready', message: 'Model ready' });
      }
      return;
    }

    if (status === 'complete') {
      this.isTranscribing = false;
      this.callbacks.delete(id);
      cb.resolve(result);
      return;
    }

    if (status === 'error') {
      this.isTranscribing = false;
      this.callbacks.delete(id);
      cb.reject(new Error(error || 'Transcription failed'));
      return;
    }
  }

  /**
   * Decode any audio File/Blob to 16,000 Hz Mono Float32Array
   */
  async decodeAudio(fileOrBlob) {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx({ sampleRate: 16000 });

    try {
      const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      
      // Resample and convert to mono 16kHz
      const targetSampleRate = 16000;
      let monoData;

      if (decodedBuffer.numberOfChannels === 1 && decodedBuffer.sampleRate === targetSampleRate) {
        monoData = decodedBuffer.getChannelData(0);
      } else {
        // OfflineAudioContext for clean resampling to 16000Hz mono
        const duration = decodedBuffer.duration;
        const offlineCtx = new OfflineAudioContext(1, Math.ceil(duration * targetSampleRate), targetSampleRate);
        const source = offlineCtx.createBufferSource();
        source.buffer = decodedBuffer;
        source.connect(offlineCtx.destination);
        source.start(0);
        const renderedBuffer = await offlineCtx.startRendering();
        monoData = renderedBuffer.getChannelData(0);
      }

      await audioCtx.close();
      return monoData;
    } catch (err) {
      try { await audioCtx.close(); } catch(e){}
      throw new Error('Failed to decode audio file. Please ensure it is a valid audio format: ' + err.message);
    }
  }

  /**
   * Transcribe an audio file or blob locally
   */
  async transcribe(fileOrBlob, options = {}) {
    if (!this.isSupported()) {
      throw new Error('Local browser transcription is not supported on this browser. Please use a modern Chrome, Edge, Safari, or Firefox browser.');
    }

    if (this.isTranscribing) {
      throw new Error('Another transcription is currently in progress. Please wait or cancel.');
    }

    const { language = 'en-IN', onProgress = null } = options;
    const taskId = ++this.currentTaskId;
    this.isTranscribing = true;

    if (onProgress) {
      onProgress({ status: 'preparing', message: 'Decoding audio file locally...' });
    }

    // Step 1: Decode audio to 16kHz mono Float32Array
    const audioFloat32 = await this.decodeAudio(fileOrBlob);

    if (!audioFloat32 || audioFloat32.length === 0) {
      this.isTranscribing = false;
      throw new Error('Audio file is empty or could not be decoded.');
    }

    // Step 2: Post to Web Worker
    return new Promise((resolve, reject) => {
      this.callbacks.set(taskId, {
        resolve: (rawResult) => {
          const formatted = this.formatDialogue(rawResult);
          resolve(formatted);
        },
        reject,
        onProgress
      });

      this.worker.postMessage({
        action: 'transcribe',
        id: taskId,
        audio: audioFloat32,
        language: language
      });
    });
  }

  /**
   * Format Whisper output chunks into conversational speaker lines
   */
  formatDialogue(rawResult) {
    const fullText = (rawResult.text || '').trim();
    const chunks = rawResult.chunks || [];
    let dialogue = [];

    if (chunks.length > 0) {
      let currentTimeOffset = 0;
      for (const chunk of chunks) {
        const text = (chunk.text || '').trim();
        if (!text) continue;
        const [start, end] = chunk.timestamp || [currentTimeOffset, currentTimeOffset + 5];
        const isAgent = /recovery|rbi|due|emi|pay|bhejo|loan|agent|telecaller|police|court|notice/i.test(text);
        dialogue.push({
          speaker: isAgent ? 'Recovery Agent' : 'Borrower / Customer',
          role: isAgent ? 'agent' : 'user',
          time: `0:${Math.floor(start).toString().padStart(2, '0')} - 0:${Math.floor(end).toString().padStart(2, '0')}`,
          text: text
        });
        currentTimeOffset = Math.floor(end);
      }
    }

    // If chunks weren't separated into dialogue turns, generate from lines/sentences
    if (dialogue.length === 0 && fullText) {
      const sentences = fullText.split(/(?<=[.?!।])\s+/).filter(s => s.trim());
      let offset = 0;
      for (let i = 0; i < sentences.length; i++) {
        const s = sentences[i].trim();
        const isAgent = (i % 2 === 0);
        dialogue.push({
          speaker: isAgent ? 'Caller / Agent' : 'Borrower / User',
          role: isAgent ? 'agent' : 'user',
          time: `0:${offset.toString().padStart(2, '0')} - 0:${(offset + 8).toString().padStart(2, '0')}`,
          text: s
        });
        offset += 9;
      }
    }

    // Generate formatted transcript string for textarea
    let transcriptText = '';
    if (dialogue.length > 0) {
      transcriptText = dialogue.map(d => `${d.speaker}: ${d.text}`).join('\n');
    } else {
      transcriptText = fullText;
    }

    return {
      status: 'success',
      source: 'local_whisper_browser',
      transcript: transcriptText,
      dialogue: dialogue,
      fullText: fullText
    };
  }

  cancel() {
    if (this.isTranscribing) {
      this.isTranscribing = false;
      this.callbacks.clear();
      this.initWorker();
      return true;
    }
    return false;
  }
}

// Attach to window
window.LocalWhisperTranscriber = new LocalWhisperTranscriber();
