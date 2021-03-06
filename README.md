## World Cup 2018

Goal: Predict the numbers of goals for England and Sweden in their upcoming game this Saturday in World Cup 2018 (Quarter-finals) using ML.NET.

Most recent program output for choosing the best algorithm:
``` ini
The best models are:
FastTreeRegressor
-- numLeaves : 5
-- learningRates : 0.1
-- maxBins : 20
-- England : 1.180031
-- Sweden : 0.7405879
-- loss : 1.25614987313573
```
Program output for choosing the best hyperparameter set within an algorithm:
``` ini
Best fast tree regressor so far is: NumLeaves = 5, LearningRates = 0.1, MaxBins = 20, with loss (RMS) 1.25614987313573
-- Predicted result is: England VS Sweden = 1.180031 : 0.7405879

Best stochastic dual coordinate ascent regressor so far is: BiasLearningRate = 0.01, L1Threshold = 10, L2Const = 0.001, with loss (RMS) 1.27173625236532
-- Predicted result is: England VS Sweden = 1.360686 : 1.070364
```

Program output without chossing the best model:
``` ini
RMS = 1.29679007132815
RSquared = -0.0500067219882872
Predicted number of goals for England: 1.102681
Predicted number of goals for Sweden: 1.346095
```
