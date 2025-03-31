A user interface to listen to a rave prior and alter latent dimensions in real time.

RAVE is a VAE (variational auto encoder) capable of generating high quality synthesized audio in real time. A prior is a learned statistical distribution from which allows for high quality autoregressive synthesis of coherent sounds similar to those of the training data.

To run: 

1. Make sure you have a RAVE model (with a prior available installed) in the working directory.
   They can be found here: https://acids-ircam.github.io/rave_models_download
2. $pip install -r requirements.txt
3. $python ui.py


