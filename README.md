# DualSPHysics+: An enhanced DualSPHysics with improvements in accuracy, energy conservation and resolution of the continuity equation.
More detail on the enhanced model can be referred to https://www.sciencedirect.com/science/article/pii/S0010465524003126

Each case contains two folders, one for code and one for configuration.
Users can use or not use the schemes, such as like "**VEM**", "**VCS**", etc., by modifying the Casedef.xml file in the configuration folder.
Simulations with different resolutions can also be carried out by modifying "_dp_" in xml file.

Users are required to recompile the program before execution, as only the source code has been provided. A recommended approach is to download the full V5.2 package from DualSPHysics official website(https://dual.sphysics.org/) and replace its source file with the current one. Note that due to subtle differences between different versions of DualSPHysics, we recommend using **DualSPHysics V5.2_BETA** for CPU code compilation and **DualSPHysics V5.2.2** for GPU code compilation.

