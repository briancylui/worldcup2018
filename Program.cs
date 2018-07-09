﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace WorldCup
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train_augmented.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test_augmented.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static readonly double EnglandGoals = 2.0;
        static readonly double SwedenGoals = 0.0;

        static readonly List<int> numLeavesList = new List<int>() { 2, 5, 10, 20 };
        static readonly List<double> learningRatesList = new List<double>() { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 };
        static readonly List<int> maxBinsList = new List<int>() { 2, 5, 10, 20 };

        static async Task Main(string[] args)
        {
            double best_loss = Double.MaxValue;

            foreach (int numLeaves in numLeavesList)
            {
                foreach (double learningRates in learningRatesList)
                {
                    foreach (int maxBins in maxBinsList)
                    {
                        var model = await Train(numLeaves, learningRates, maxBins);
                        double loss = Evaluate(model);
                        WorldCupPrediction predictionForEngland = model.Predict(TestData.TestEngland);
                        WorldCupPrediction predictionForSweden = model.Predict(TestData.TestSweden);

                        if (loss < best_loss)
                        {
                            Console.WriteLine($"Best model so far is: NumLeaves = {numLeaves}, LearningRates = {learningRates}, MaxBins = {maxBins}, with loss {loss}");
                            Console.WriteLine($"-- Predicted result is: England VS Sweden = {predictionForEngland.HomeTeamGoals} : {predictionForSweden.HomeTeamGoals}");
                            best_loss = loss;
                        }
                    }
                }
            }


        }

        public static async Task<PredictionModel<WorldCupData, WorldCupPrediction>> Train(int numLeaves, double learningRates, int maxBins)
        {
            var pipeline = new LearningPipeline()
            {
                new TextLoader(_dataPath).CreateFrom<WorldCupData>(useHeader: true, separator: ','),
                new ColumnCopier(("HomeTeamGoals", "Label")),
                new CategoricalOneHotVectorizer(
                    "Stage",
                    "HomeTeam",
                    "AwayTeam",
                    "Referee"),
                new ColumnConcatenator(
                    "Features",
                    "Year",
                    "Stage",
                    "HomeTeam",
                    "AwayTeam",
                    "Attendance",
                    "Referee"),
                new FastTreeRegressor() { NumLeaves = numLeaves, LearningRates = learningRates, MaxBins = maxBins }
            };

            PredictionModel<WorldCupData, WorldCupPrediction> model = pipeline.Train<WorldCupData, WorldCupPrediction>();

            await model.WriteAsync(_modelPath);
            return model;
        }

        public static double Evaluate(PredictionModel<WorldCupData, WorldCupPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<WorldCupData>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            return metrics.Rms;
        }
    }
}
