# Variational Inference for Deep State-Space Model

This repository explores latent dynamics modeling for MDP/PODMP with pixel observations using deep state-space models. We are particularly interested in the case where the dynamics is sufficiently stochastic and the next state distribution $p(s^\prime \mid s,a)$ is potentially multi-modal.

![graphical model](https://github.com/user-attachments/assets/3162d585-dccb-47e4-a4ae-0ef4225d2240)

For evaluation, we devise a variation of the toy benchmark from SV2P [^1].

[^1]: Babaeizadeh, Mohammad, et al. "Stochastic Variational Video Prediction." _International Conference on Learning Representations_. 2018.
