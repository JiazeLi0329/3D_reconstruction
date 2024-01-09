# 3 Dimensional Reconstruction
3 dimensional reconstruction of the human body in military sports

## 3D_no_grid.py
3D_no_grid file does not render the floor and wall grids in the 3D reconstruction, highlighting the human skeleton pose.

## 3D_fixed_grid.py
3D_fixed_grid file fixes the floor and walls during 3D reconstruction, and sets the velocity of the right foot to zero for the rendering of the human posture.

## 3D_dynamic_grid.py
3D_dynamic_grid file calculates the floor plane using the RANSAC algorithm during 3D reconstruction and renders the image based on the parameters obtained.
