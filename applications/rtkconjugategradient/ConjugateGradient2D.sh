# Generate geometry for fan-beam setup
rtksimulatedgeometry -n 720 -o geometry.xml --arc 360

# Create projections of the phantom file
# Note the sinogram being 3 pixels wide in the y direction to allow back-projection interpolation in a 2D image
rtkprojectgeometricphantom -g geometry.xml -o projections.mha --spacing 0.5 --dimension 1024,3,720 --phantomfile SheppLogan-2d.txt

# Perform Conjugate Gradient reconstruction
rtkconjugategradient -p . -r projections.mha -o cg.mha -g geometry.xml --spacing 0.5 --dimension 512 3 512 -n 50

# Create a reference volume for comparison
rtkdrawgeometricphantom --spacing 0.5 --dimension 512,1,512 -o initial_volume.mha --phantomscale=512,1,512 --phantomfile SheppLogan-2d.txt