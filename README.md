# AI_ART
• An evolutionary algorithm was designed to reproduce and generate images in an artistic way. The algorithm
is given a reference image and then tries to regenerate this image using polygons and different shapes. The
algorithm was designed and implemented in python which has some benefits in terms of the available tools
for image manipulation but also has some drawbacks in term of run-time speed in pixel rendering and canvas
allocation.
• The main idea of the algorithm is to try approaching the original image by drawing shapes on the canvas and
try to improve these drawings generation after another depending on the value of the fitness function. The
algorithm starts from random N polygons with some transparency ratio. In each optimization step it randomly
modifies one parameter (like color components or polygon vertices) and check whether such new variant looks
more like the original image. If it is, we keep it, and continue to mutate this one instead. The final image is
represented as a collection of overlapping polygons of various colors and transparencies to get an artistic view
of the source image.
