## Outline

- clearly demonstrate how DFT/FFT works
  - explain the difficult parts
    - aliasing
    - leakage
    - ...

- data over sound
  - decode ggwave data
    - look at the difference of 8kHz, 16kHz and 32kHz sampled signal FFT and 44.1kHz

- FIR filters

- generate an animation of a Fourier Transform window scrolling over a sound file. Best would be to do that on the ggwave sound.
  - improve the plotting function
    - I think the simple example from fft.py has a more robust plotting, move it to dos_fft_cli.py
    - narrow the frequency bin space to frequencies of interest
