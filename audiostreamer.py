import numpy as np
import sounddevice as sd
import time
import threading
import queue
import torch

model = torch.jit.load('musicnet.ts')

class AudioStreamer:
    """
    Class to continuously stream audio that's generated in chunks
    """
    def __init__(self, sample_rate=44100, buffer_size=2048, max_queue_size=32):
        """
        Initialize the audio streamer
        
        Parameters:
        -----------
        sample_rate : int
            Sample rate of the audio (default: 44100 Hz)
        buffer_size : int
            Size of audio blocks to process at a time (default: 2048)
        max_queue_size : int
            Maximum number of chunks to buffer (default: 32)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.is_active = False
        self.stream = None
    
    def add_chunk(self, audio_chunk):
        """
        Add a new chunk of audio to the streaming queue
        
        Parameters:
        -----------
        audio_chunk : numpy.ndarray
            New chunk of audio to be streamed (should be of length buffer_size)
        
        Returns:
        --------
        bool : Whether the chunk was successfully added to the queue
        """
        if self.stop_event.is_set():
            return False
            
        # Ensure audio is float32 and properly normalized
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val
        
        # Ensure correct size
        if len(audio_chunk) != self.buffer_size:
            # Resize if needed - either truncate or pad with zeros
            new_chunk = np.zeros(self.buffer_size, dtype=np.float32)
            copy_size = min(len(audio_chunk), self.buffer_size)
            new_chunk[:copy_size] = audio_chunk[:copy_size]
            audio_chunk = new_chunk
        
        # Try to add to queue, but don't block for too long
        try:
            self.audio_queue.put(audio_chunk, block=True, timeout=0.1)
            return True
        except queue.Full:
            print("Warning: Audio buffer full, dropping chunk")
            return False
    
    def audio_callback(self, outdata, frames, time, status):
        """Callback for sounddevice to get audio data"""
        if status:
            print(f"Audio status: {status}")
        
        try:
            if not self.audio_queue.empty():
                data = self.audio_queue.get_nowait()
                if len(data) == frames:
                    outdata[:] = data.reshape(-1, 1)  # Mono to stereo conversion
                else:
                    outdata[:len(data)] = data.reshape(-1, 1)
                    outdata[len(data):] = 0
            else:
                outdata.fill(0)  # Fill with silence if queue is empty
                print("Buffer underrun - audio queue is empty!")
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata.fill(0)
    
    def start(self):
        """Start the audio stream"""
        if self.is_active:
            print("Audio stream is already active")
            return False
            
        # Reset stop event
        self.stop_event.clear()
        
        # Initialize and start the stream
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,  # Mono output
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        
        self.stream.start()
        self.is_active = True
        print(f"Audio streaming started at {self.sample_rate} Hz with buffer size {self.buffer_size}")
        return True
    
    def stop(self):
        """Stop the audio stream"""
        if not self.is_active:
            return
            
        # Set stop event and clean up
        self.stop_event.set()
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass
        
        # Stop and close the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.is_active = False
        print("Audio streaming stopped")
    
    def get_queue_status(self):
        """Return the current queue size and capacity"""
        return {
            "current_size": self.audio_queue.qsize(),
            "capacity": self.audio_queue.maxsize,
            "fullness_percent": (self.audio_queue.qsize() / self.audio_queue.maxsize) * 100
        }
    
    def __del__(self):
        """Ensure resources are released when object is deleted"""
        self.stop()


# Example usage for continuous generation in 2048-sample chunks
if __name__ == "__main__":
    import numpy as np
    
    # Parameters
    sample_rate = 44100  # Hz
    buffer_size = 2048   # Samples per chunk
    duration = 10        # Seconds to run the demo
    
    # Create the audio streamer
    streamer = AudioStreamer(sample_rate=sample_rate, buffer_size=buffer_size)
    
    # Start the stream
    streamer.start()
    
    # Function that simulates generating audio chunks (replace with your actual generator)
    def generate_chunk():
        with torch.no_grad():
            latent_vector = torch.FloatTensor(1, 8, 1).uniform_(-3, 0) 
            audio = model.decode(latent_vector)
            audio = audio.squeeze().cpu().numpy()
        return audio
    
    try:
        start_time = time.time()            
        t = 0  # Time tracker
         
        # Main loop - would be replaced by your actual generation code
        while time.time() - start_time < duration:
            # Generate a new chunk
            chunk = generate_chunk()
            
            # Add to the streamer
            if not streamer.add_chunk(chunk):
                print("Failed to add chunk - stream may have stopped")
                break
                
            # Update time tracker
            t += buffer_size / sample_rate
                
            # Check queue status occasionally
            #if int(t) % 1 == 0:  # Every second
                #status = streamer.get_queue_status()
                #print(f"Queue status: {status['current_size']}/{status['capacity']} chunks " +
                      #f"({status['fullness_percent']:.1f}% full)")
            
            # Sleep a small amount to simulate generation time
            # In real usage, your generation code would naturally take some time
            
    except KeyboardInterrupt:
        print("Streaming interrupted by user")
    finally:
        # Clean up
        streamer.stop()
        print("Demo finished")