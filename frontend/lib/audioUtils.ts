/**
 * Audio utility functions for converting and encoding audio
 */

/**
 * Convert an audio blob to WAV format
 * @param audioBlob - Input audio blob (any format supported by browser)
 * @returns WAV format blob
 */
export async function convertToWav(audioBlob: Blob): Promise<Blob> {
  // Create an audio context
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();

  // Read the blob as array buffer
  const arrayBuffer = await audioBlob.arrayBuffer();

  // Decode the audio data
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Convert to WAV
  const wavBlob = audioBufferToWav(audioBuffer);

  // Close the audio context to free resources
  await audioContext.close();

  return wavBlob;
}

/**
 * Convert an AudioBuffer to WAV blob
 * @param audioBuffer - Web Audio API AudioBuffer
 * @returns WAV format blob
 */
function audioBufferToWav(audioBuffer: AudioBuffer): Blob {
  const numberOfChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;

  // Interleave channels
  const length = audioBuffer.length * numberOfChannels * 2; // 2 bytes per sample (16-bit)
  const buffer = new ArrayBuffer(44 + length);
  const view = new DataView(buffer);

  // Write WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, format, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * bitDepth / 8, true); // byte rate
  view.setUint16(32, numberOfChannels * bitDepth / 8, true); // block align
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, length, true);

  // Write audio data
  const channels: Float32Array[] = [];
  for (let i = 0; i < numberOfChannels; i++) {
    channels.push(audioBuffer.getChannelData(i));
  }

  let offset = 44;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Write a string to a DataView at a given offset
 */
function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}
