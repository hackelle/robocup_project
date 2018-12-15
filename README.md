# Robocup Project: Nao-EMOM

[SS18] PJ Rob
Project Robocup
TU Berlin

## Collaborators:
- Hackel, Leonard
- Mengers, Vito
- v. Blanckenburg, Jasper 

## Project goal:

Detect the direction other robots are looking in:

old:
- Find the eyes (good contrast due to LEDs)
- Find the head and its dimensions
- Calculate robot distance based on head height
- Calculate angle from robot distance and eye distance (should give 1/2 options)
- Select correct angle based on ears

new:
- Find the head and its approximate dimensions using Tensorflow
- Find Ellipses, devide them into possible ears and eyes
- Calculate robot distance based on ear height
- Calculate head angle based on ear width, position of eyes relative to the ear and if we see eyes at all
