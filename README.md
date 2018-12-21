# Robocup Project: Nao EMOM

[SS18] PJ Rob
Project Robocup
TU Berlin

## Collaborators
- Hackel, Leonard
- Mengers, Vito
- v. Blanckenburg, Jasper

## Dependencies
- PyQT 5
- Python TensorFlow
- TensorFlow Object Detection
- Python 2.7
- numpy
- OpenCV

## Project Structure
The main python scripts are located in the `project` directory. All other directories contain additonal data like training data for the CNN.

## Usage
```bash
project/debug_GUI.py {rt,img,dir} location
```

The first argument determines the type of image provider:

- `rt` takes realtime images from the robot at the IP in `location`
- `img` uses a single image stored at `location`
- `dir` uses all images in the directory at `location`

## Report
A description of the algorithm can be found in [Nao_EMOM.pdf](Nao_EMOM.pdf).
