# InpaintAR

This project contains the Unity Project and the LaTeX Thesis of my Bachelors Thesis about Inpainting Methods in Realtime for XR.

The Project is still Work in Progress, the following milestones are planned:

The Following should be achieved through the Passthrough API
- Selection of Planes with the controller
- Detection of Objects on top of the planes

The following will be implemented on top of the before mentioned features
- Covering of the Objects on the planes through virtual objects with a fixed color
- Implementation of the usage of the camera pixels to texture said virtual object so it is not visible
- Implementation of an initial Inpainting Algorithm to apply to the texture instead of the existing camera pixels
- Trial with virtual Objects to try the behavior (probably use Depth API of Meta)