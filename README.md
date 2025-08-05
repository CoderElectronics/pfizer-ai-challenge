# pfizer-ai-challenge

My finalist solution to the Pfizer x DRL AI Challenge. This was and is nothing more than a very simple implementation of basic OpenCV tracking by comparing changes in frames, due to the fact that the competition provided largely static perspective videos.

My second attempt used a combination of image sharpening and upscaling techniques combined with basic CNNs and rule-based 2D spline interpolation to predict the path of the drone in all cases, even when occluded by other elements, but it simply wasn't reliable enough submit by the deadline.

![Video of CV tracking in action](docs/cvtracking.gif)

# Challenge Description
The challenge presented to the participants is to create a mechanism of autonomously tracking a drone in a video taken within the DRL racing simulator.
Contestants will be provided with a practice video of a flight on a game map, practice input timestamps, and practice ground truth data in CSV format on which to train/refine their code. The contestants will submit the results of their development in the form of a zip file and will be scored on the best performance of their code applied to a “scoring” video that captures a different flight on the same game map.
