# PitchPredict
### PitchPredict is an ML model which tries to predict the next pitch in an at-bat, based on data about that at-bat.
PitchPredict was trained on every pitch from the start of the 2025 MLB season to today, about 1.2 million in total.  
PitchPredict has an accuracy of about 45%, which beats guessing the most common pitch every time by about 10%. In my research, this seemed to be the limit for a general-purpose model of this type. Top-3 accuracy is about 85%.  
PitchPredict was created using pytorch, and uses a neural net architecture with a few embeddings for features such as pitcher/batter identity, previous pitches, and inning state. It also uses linear and reLU layers for numeric features, such as batter slash line and game state.  
Generative AI was used to create PitchPredict, primarily in writing data gathering code, but also on occasion for code seen in model.py.  
PitchPredict has a web interface accessible [here](https://epshteinmatthew.github.io/PitchPredictWeb/). It predicts the next pitch in every live MLB at-bat and tracks the accuracy of those predictions. Screenshots forthcoming.
