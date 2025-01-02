# Models

This folder contains model implementations for baseline and newly proposed works. Supported models for 3D object 
detection and visual grounding currently include
* OpenScene
* LERF (through `nerfstudio`)

Additionally, other models may be supported here as auxiliary models or sub-modules of other models. This list includes
* CLIP

While in most cases we will opt to import existing libraries for these models as appropriate, we will occasionally 
need to implement wrapper classes here to make them compatible with our codebase.

Note also that, for baseline models, we largely leave the training of these models to their individual repositories
and primarily focus on supporting standardized inference and evaluation here.