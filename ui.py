import tkinter as tk
from tkinter import ttk
import threading
import time
import torch
import numpy as np
import logging
import queue

# Import our custom modules
from latent import LatentController
from audiostreamer import AudioStreamer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LatentUI')

class LatentControlUI:
    """
    UI for controlling latent dimensions (bias and scale) and streaming audio
    """
    def __init__(self, model_path='musicnet.ts', sample_rate=44100, buffer_size=2048):
        """
        Initialize the latent control UI
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
        sample_rate : int
            Sample rate for audio output
        buffer_size : int
            Size of audio buffer chunks
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Load the model
        try:
            self.model = torch.jit.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.model = None
            logger.error(f"Failed to load model: {e}")
        
        # Create latent controller
        self.latent_controller = LatentController(
            model=self.model,
            min_bias=-3.0,
            max_bias=3.0,
            min_scale=0.0,
            max_scale=1.0
        )
        
        # Create audio streamer
        self.audio_streamer = AudioStreamer(sample_rate=sample_rate, buffer_size=buffer_size)
        
        # Initialize UI components
        self.root = None
        self.bias_sliders = []
        self.scale_sliders = []
        self.bias_values = []
        self.scale_values = []
        self.base_values = []
        self.is_running = False
        self.audio_thread = None
        self.audio_playing = False
        
        # Build the UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the Tkinter UI"""
        self.root = tk.Tk()
        self.root.title("Latent Audio Control - Bias & Scale")
        
        # Set window size and position
        window_width = 800
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Add a title label
        title_label = ttk.Label(self.root, text="Latent Dimension Control - Bias & Scale", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # Create a main frame with a canvas for scrolling
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Frame for the controls
        controls_frame = ttk.Frame(scrollable_frame)
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create header labels
        header_frame = ttk.Frame(controls_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        dim_label = ttk.Label(header_frame, text="Dimension", width=10)
        dim_label.pack(side=tk.LEFT, padx=5)
        
        base_label = ttk.Label(header_frame, text="Base Value", width=10)
        base_label.pack(side=tk.LEFT, padx=5)
        
        bias_label = ttk.Label(header_frame, text="Bias (-3 to 3)", width=15)
        bias_label.pack(side=tk.LEFT, padx=5)
        
        scale_label = ttk.Label(header_frame, text="Scale (0 to 1)", width=15)
        scale_label.pack(side=tk.LEFT, padx=5)
        
        final_label = ttk.Label(header_frame, text="Final Value", width=10)
        final_label.pack(side=tk.LEFT, padx=5)
        
        # Create row for each latent dimension
        for i in range(self.latent_controller.latent_dims):
            row_frame = ttk.Frame(controls_frame)
            row_frame.pack(fill=tk.X, pady=5)
            
            # Dimension label
            dim_label = ttk.Label(row_frame, text=f"Dim {i+1}", width=10)
            dim_label.pack(side=tk.LEFT, padx=5)
            
            # Base value label
            base_var = tk.StringVar(value=f"{self.latent_controller.get_base_value(i):.2f}")
            self.base_values.append(base_var)
            base_value_label = ttk.Label(row_frame, textvariable=base_var, width=10)
            base_value_label.pack(side=tk.LEFT, padx=5)
            
            # Bias slider and value
            bias_frame = ttk.Frame(row_frame)
            bias_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            bias_var = tk.StringVar(value="0.00")
            self.bias_values.append(bias_var)
            
            bias_slider = ttk.Scale(
                bias_frame, 
                from_=self.latent_controller.min_bias, 
                to=self.latent_controller.max_bias,
                orient=tk.HORIZONTAL,
                command=lambda val, idx=i: self._on_bias_slider_changed(val, idx)
            )
            bias_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.bias_sliders.append(bias_slider)
            
            bias_value_label = ttk.Label(bias_frame, textvariable=bias_var, width=6)
            bias_value_label.pack(side=tk.LEFT, padx=2)
            
            # Scale slider and value
            scale_frame = ttk.Frame(row_frame)
            scale_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            scale_var = tk.StringVar(value="1.00")
            self.scale_values.append(scale_var)
            
            scale_slider = ttk.Scale(
                scale_frame, 
                from_=self.latent_controller.min_scale, 
                to=self.latent_controller.max_scale,
                orient=tk.HORIZONTAL,
                command=lambda val, idx=i: self._on_scale_slider_changed(val, idx)
            )
            scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.scale_sliders.append(scale_slider)
            
            scale_value_label = ttk.Label(scale_frame, textvariable=scale_var, width=6)
            scale_value_label.pack(side=tk.LEFT, padx=2)
            
            # Final value label (base * scale + bias)
            final_var = tk.StringVar(value=f"{self.latent_controller.current_vector[0, i, 0].item():.2f}")
            final_label = ttk.Label(row_frame, textvariable=final_var, width=10)
            final_label.pack(side=tk.LEFT, padx=5)
            
            # Set initial slider values
            bias_slider.set(self.latent_controller.get_bias(i))
            scale_slider.set(self.latent_controller.get_scale(i))
            
            # Update text values
            self._update_dimension_values(i)
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, pady=10, padx=20)
        
        # Create buttons
        self.play_pause_btn = ttk.Button(button_frame, text="Play Audio", 
                                          command=self._toggle_audio)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        interpolate_btn = ttk.Button(button_frame, text="Interpolate", 
                                      command=self._start_interpolation)
        interpolate_btn.pack(side=tk.LEFT, padx=5)
        
        reset_bias_btn = ttk.Button(button_frame, text="Reset Bias", 
                                     command=self._reset_biases)
        reset_bias_btn.pack(side=tk.LEFT, padx=5)
        
        reset_scale_btn = ttk.Button(button_frame, text="Reset Scale", 
                                      command=self._reset_scales)
        reset_scale_btn.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=5, padx=20)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        # Set the protocol for when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _update_dimension_values(self, idx):
        """Update all displayed values for a given dimension"""
        try:
            # Update base value display
            base_val = self.latent_controller.get_base_value(idx)
            self.base_values[idx].set(f"{base_val:.2f}")
            
            # Update bias value display
            bias_val = self.latent_controller.get_bias(idx)
            self.bias_values[idx].set(f"{bias_val:.2f}")
            
            # Update scale value display
            scale_val = self.latent_controller.get_scale(idx)
            self.scale_values[idx].set(f"{scale_val:.2f}")
        except Exception as e:
            logger.error(f"Error updating dimension values: {e}")
    
    def _on_bias_slider_changed(self, value, idx):
        """Handle bias slider value changes"""
        try:
            value = float(value)
            self.latent_controller.set_bias(idx, value)
            self.bias_values[idx].set(f"{value:.2f}")
            
            # No need to regenerate audio here as the audio thread will 
            # regularly fetch the latest latent vector
        except Exception as e:
            logger.error(f"Error updating bias value: {e}")
    
    def _on_scale_slider_changed(self, value, idx):
        """Handle scale slider value changes"""
        try:
            value = float(value)
            self.latent_controller.set_scale(idx, value)
            self.scale_values[idx].set(f"{value:.2f}")
        except Exception as e:
            logger.error(f"Error updating scale value: {e}")
    
    def _toggle_audio(self):
        """Toggle audio playback on/off"""
        if self.audio_playing:
            # Stop audio
            self.audio_playing = False
            self.play_pause_btn.configure(text="Play Audio")
            self.audio_streamer.stop()
            self.status_var.set("Audio paused")
            logger.info("Audio playback stopped")
        else:
            # Start audio
            self.audio_playing = True
            self.play_pause_btn.configure(text="Stop Audio")
            self.audio_streamer.start()
            self.status_var.set("Audio playing")
            logger.info("Audio playback started")
    
    def _reset_biases(self):
        """Reset all biases to zero"""
        try:
            # Reset all biases
            self.latent_controller.biases = torch.zeros(self.latent_controller.latent_dims)
            
            # Recompute the current vector
            self.latent_controller.current_vector = self.latent_controller._compute_current_vector()
            
            # Update all sliders and values
            for i in range(self.latent_controller.latent_dims):
                self.bias_sliders[i].set(0.0)
                self.bias_values[i].set("0.00")
            
            self.status_var.set("Reset all biases to zero")
        except Exception as e:
            logger.error(f"Error resetting biases: {e}")
            self.status_var.set(f"Error: {e}")
    
    def _reset_scales(self):
        """Reset all scales to one"""
        try:
            # Reset all scales
            self.latent_controller.scales = torch.ones(self.latent_controller.latent_dims)
            
            # Recompute the current vector
            self.latent_controller.current_vector = self.latent_controller._compute_current_vector()
            
            # Update all sliders and values
            for i in range(self.latent_controller.latent_dims):
                self.scale_sliders[i].set(1.0)
                self.scale_values[i].set("1.00")
            
            self.status_var.set("Reset all scales to one")
        except Exception as e:
            logger.error(f"Error resetting scales: {e}")
            self.status_var.set(f"Error: {e}")
    
    def _start_interpolation(self):
        """Start interpolation to random target biases and scales"""
        try:
            # Generate random targets for biases and scales
            target_biases = torch.FloatTensor(self.latent_controller.latent_dims).uniform_(
                self.latent_controller.min_bias,
                self.latent_controller.max_bias
            )
            
            target_scales = torch.FloatTensor(self.latent_controller.latent_dims).uniform_(
                self.latent_controller.min_scale,
                self.latent_controller.max_scale
            )
            
            steps = 50  # 50 interpolation steps
            
            if self.latent_controller.start_interpolation(target_biases, target_scales, steps):
                self.status_var.set("Interpolation started")
                
                # Start a thread to update sliders during interpolation
                threading.Thread(target=self._interpolation_update_thread, daemon=True).start()
        except Exception as e:
            logger.error(f"Error starting interpolation: {e}")
            self.status_var.set(f"Error: {e}")
    
    def _interpolation_update_thread(self):
        """Thread to update UI during interpolation"""
        while self.latent_controller.interpolation_active and self.is_running:
            # Update all slider positions based on current values
            for i in range(self.latent_controller.latent_dims):
                # Update bias slider
                bias_val = self.latent_controller.get_bias(i)
                self.root.after(0, lambda slider=self.bias_sliders[i], val=bias_val: slider.set(val))
                self.root.after(0, lambda var=self.bias_values[i], val=bias_val: var.set(f"{val:.2f}"))
                
                # Update scale slider
                scale_val = self.latent_controller.get_scale(i)
                self.root.after(0, lambda slider=self.scale_sliders[i], val=scale_val: slider.set(val))
                self.root.after(0, lambda var=self.scale_values[i], val=scale_val: var.set(f"{val:.2f}"))
            
            # Sleep a bit to not overload the UI
            time.sleep(0.05)
    
    def _audio_generation_thread(self):
        """Thread to continuously generate audio from latent vectors"""
        logger.info("Audio generation thread started")
        while self.is_running:
            try:
                # Step interpolation if active
                if self.latent_controller.interpolation_active:
                    self.latent_controller.step_interpolation()
                
                # Generate audio from current latent vector
                if self.audio_playing:
                    audio = self.latent_controller.generate_audio()
                    
                    # Add to the audio streamer
                    if not self.audio_streamer.add_chunk(audio):
                        logger.warning("Failed to add audio chunk to streamer")
                
                # Sleep a little to prevent CPU overload
                # The exact amount depends on your buffer size and sample rate
                sleep_time = self.buffer_size / self.sample_rate / 2
                time.sleep(sleep_time)
            
            except Exception as e:
                logger.error(f"Error in audio generation thread: {e}")
                time.sleep(0.1)  # Prevent tight loop in case of errors
    
    def _on_close(self):
        """Handle window close event"""
        self.is_running = False
        
        # Stop any active audio
        self.audio_streamer.stop()
        
        # Wait for audio thread to end
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        # Destroy the root window
        self.root.destroy()
        logger.info("UI closed and resources released")
    
    def start(self):
        """Start the UI and audio processing"""
        if self.model is None:
            self.status_var.set("Error: No model loaded")
            logger.error("Cannot start: No model loaded")
            return
        
        self.is_running = True
        
        # Start audio generation thread (but not audio playback yet)
        self.audio_thread = threading.Thread(target=self._audio_generation_thread, daemon=True)
        self.audio_thread.start()
        
        # Start UI main loop
        self.status_var.set("Ready - Press 'Play Audio' to start")
        self.root.mainloop()

if __name__ == "__main__":
    # Create and start the UI
    ui = LatentControlUI()
    ui.start()