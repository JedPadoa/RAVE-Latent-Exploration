import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LatentController')

class LatentController:
    """
    Class to generate and manipulate latent vectors for audio generation.
    Enables control of individual dimensions in the latent space with both bias and scale.
    """
    def __init__(self, latent_dims=16, model=None, 
                 min_bias=-3.0, max_bias=3.0, 
                 min_scale=0.0, max_scale=1.0):
        """
        Initialize the latent controller
        
        Parameters:
        -----------
        latent_dims : int
            Number of dimensions in the latent space (default: 8)
        model : torch.jit.ScriptModule
            The model to use for decoding (optional)
        min_bias : float
            Minimum value for latent dimension bias (default: -3.0)
        max_bias : float
            Maximum value for latent dimension bias (default: 3.0)
        min_scale : float
            Minimum value for latent dimension scale (default: 0.0)
        max_scale : float
            Maximum value for latent dimension scale (default: 1.0)
        """
        self.latent_dims = latent_dims
        self.model = model
        
        # Bias control range
        self.min_bias = min_bias
        self.max_bias = max_bias
        
        # Scale control range
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Initialize a random base vector (original values before scaling/bias)
        self.base_vector = self._generate_random_base_vector()
        
        # Initialize bias and scale for each dimension
        self.biases = torch.zeros(self.latent_dims)
        self.scales = torch.ones(self.latent_dims)
        
        # Current vector is computed from base vector, scales, and biases
        self.current_vector = self._compute_current_vector()
        
        # Track interpolation state
        self.interpolation_active = False
        self.interpolation_target_biases = None
        self.interpolation_target_scales = None
        self.interpolation_steps = 0
        self.interpolation_current_step = 0
        
        logger.info(f"LatentController initialized with {latent_dims} dimensions")
    
    def _generate_random_base_vector(self):
        """Generate a random base latent vector with values between -1 and 1"""
        return torch.zeros(1, self.latent_dims, 1)
    
    def _compute_current_vector(self):
        """Compute the current vector by applying scales and biases to the base vector"""
        # Apply scale and bias to each dimension
        result = self.base_vector.clone()
        for dim in range(self.latent_dims):
            result[0, dim, 0] = self.scales[dim] * self.base_vector[0, dim, 0] + self.biases[dim]
        return result
    
    def randomize(self):
        """Generate a new random base vector and reset scales and biases"""
        self.base_vector = self._generate_random_base_vector()
        self.biases = torch.zeros(self.latent_dims)
        self.scales = torch.ones(self.latent_dims)
        self.current_vector = self._compute_current_vector()
        logger.debug("Generated new random latent vector with default scales and biases")
        return self.current_vector.clone()
    
    def get_vector(self):
        """Get the current latent vector"""
        return self.current_vector.clone()
    
    def set_bias(self, dim, value):
        """
        Set the bias for a specific dimension
        
        Parameters:
        -----------
        dim : int
            The dimension to modify (0-indexed)
        value : float
            The bias value to set for the specified dimension
        
        Returns:
        --------
        bool : Whether the operation was successful
        """
        if not 0 <= dim < self.latent_dims:
            logger.warning(f"Invalid dimension: {dim}, must be between 0 and {self.latent_dims-1}")
            return False
            
        # Clamp the value to valid range
        clamped_value = max(min(value, self.max_bias), self.min_bias)
        if clamped_value != value:
            logger.debug(f"Bias value {value} clamped to {clamped_value}")
            
        # Update the bias
        self.biases[dim] = clamped_value
        
        # Update the current vector
        self.current_vector = self._compute_current_vector()
        return True
    
    def set_scale(self, dim, value):
        """
        Set the scale for a specific dimension
        
        Parameters:
        -----------
        dim : int
            The dimension to modify (0-indexed)
        value : float
            The scale value to set for the specified dimension
        
        Returns:
        --------
        bool : Whether the operation was successful
        """
        if not 0 <= dim < self.latent_dims:
            logger.warning(f"Invalid dimension: {dim}, must be between 0 and {self.latent_dims-1}")
            return False
            
        # Clamp the value to valid range
        clamped_value = max(min(value, self.max_scale), self.min_scale)
        if clamped_value != value:
            logger.debug(f"Scale value {value} clamped to {clamped_value}")
            
        # Update the scale
        self.scales[dim] = clamped_value
        
        # Update the current vector
        self.current_vector = self._compute_current_vector()
        return True
    
    def get_bias(self, dim):
        """Get the bias of a specific dimension"""
        if not 0 <= dim < self.latent_dims:
            logger.warning(f"Invalid dimension: {dim}, must be between 0 and {self.latent_dims-1}")
            return None
        return self.biases[dim].item()
    
    def get_scale(self, dim):
        """Get the scale of a specific dimension"""
        if not 0 <= dim < self.latent_dims:
            logger.warning(f"Invalid dimension: {dim}, must be between 0 and {self.latent_dims-1}")
            return None
        return self.scales[dim].item()
    
    def get_base_value(self, dim):
        """Get the base value (before scaling/bias) of a specific dimension"""
        if not 0 <= dim < self.latent_dims:
            logger.warning(f"Invalid dimension: {dim}, must be between 0 and {self.latent_dims-1}")
            return None
        return self.base_vector[0, dim, 0].item()
    
    def start_interpolation(self, target_biases=None, target_scales=None, steps=30):
        """
        Start interpolating from current biases and scales to target values
        
        Parameters:
        -----------
        target_biases : torch.Tensor or None
            Target biases to interpolate to. If None, random biases are generated.
        target_scales : torch.Tensor or None
            Target scales to interpolate to. If None, random scales are generated.
        steps : int
            Number of steps to take for the interpolation
        """
        # Generate random targets if not provided
        if target_biases is None:
            target_biases = torch.FloatTensor(self.latent_dims).uniform_(self.min_bias, self.max_bias)
        elif len(target_biases) != self.latent_dims:
            logger.error(f"Target biases length {len(target_biases)} doesn't match expected length {self.latent_dims}")
            return False
            
        if target_scales is None:
            target_scales = torch.FloatTensor(self.latent_dims).uniform_(self.min_scale, self.max_scale)
        elif len(target_scales) != self.latent_dims:
            logger.error(f"Target scales length {len(target_scales)} doesn't match expected length {self.latent_dims}")
            return False
            
        self.interpolation_target_biases = target_biases
        self.interpolation_target_scales = target_scales
        self.interpolation_steps = steps
        self.interpolation_current_step = 0
        self.interpolation_active = True
        
        logger.info(f"Starting interpolation over {steps} steps")
        return True
    
    def step_interpolation(self):
        """
        Take one step in the current interpolation
        
        Returns:
        --------
        torch.Tensor or None : The updated vector if interpolation is active, None otherwise
        """
        if not self.interpolation_active:
            return None
            
        # Calculate progress (0.0 to 1.0)
        progress = self.interpolation_current_step / self.interpolation_steps
        
        if progress >= 1.0:
            # Interpolation complete
            self.biases = self.interpolation_target_biases.clone()
            self.scales = self.interpolation_target_scales.clone()
            self.current_vector = self._compute_current_vector()
            self.interpolation_active = False
            logger.debug("Interpolation complete")
            return self.current_vector.clone()
            
        # Linear interpolation for both biases and scales
        self.biases = self.biases + progress * (self.interpolation_target_biases - self.biases)
        self.scales = self.scales + progress * (self.interpolation_target_scales - self.scales)
        
        # Update the current vector
        self.current_vector = self._compute_current_vector()
        
        self.interpolation_current_step += 1
        return self.current_vector.clone()
    
    def stop_interpolation(self):
        """Stop any active interpolation"""
        if self.interpolation_active:
            self.interpolation_active = False
            logger.debug("Interpolation stopped")
    
    def generate_audio(self):
        """
        Generate audio from the current latent vector using the model
        
        Returns:
        --------
        numpy.ndarray : Generated audio samples
        """
        if self.model is None:
            logger.error("No model is set for audio generation")
            return np.zeros(2048, dtype=np.float32)  # Return silence
            
        try:
            with torch.no_grad():
                audio = self.model.decode(self.current_vector)
                audio = audio.squeeze().cpu().numpy()
            return audio
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return np.zeros(2048, dtype=np.float32)  # Return silence

    def set_model(self, model):
        """Set the model to use for audio generation"""
        self.model = model
        logger.info("Model updated for audio generation")
        
        
if __name__ == '__main__':
    # Test the LatentController class
    model = torch.jit.load('musicnet.ts')
    controller = LatentController(latent_dims=8)
    with torch.no_grad():
        audio = model.decode(controller.get_vector())
        audio = audio.squeeze().cpu().numpy()
    print(audio.shape)