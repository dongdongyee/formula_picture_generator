# Data Generator for Formula Detection
The code automatically generates various fake pictures for formula detection
# Usage
1. prepare some text corpus and put it to ./word4gen
2. place your font files under./font
3. place the LaTeX formula corpus to ./word4gen
```
python generate_pictures.py num_pictures output_dir num_process
```

Then you will get fake images and its json file for corresponding bounding boxes for formulas.
Here's an example: isollated formulas marked in blue and embedded formulas marked in red.
![Alt text](imgs/outputformulas_with_words_000007.png?raw=true "Title")

