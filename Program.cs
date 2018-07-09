using System;
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

        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            WorldCupPrediction predictionForEngland = model.Predict(TestData.TestEngland);
            Console.WriteLine($"Predicted number of goals for England: {predictionForEngland.HomeTeamGoals}");
            WorldCupPrediction predictionForSweden = model.Predict(TestData.TestSweden);
            Console.WriteLine($"Predicted number of goals for Sweden: {predictionForSweden.HomeTeamGoals}");
        }

        public static async Task<PredictionModel<WorldCupData, WorldCupPrediction>> Train()
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
                new FastTreeRegressor()
            };

            PredictionModel<WorldCupData, WorldCupPrediction> model = pipeline.Train<WorldCupData, WorldCupPrediction>();

            await model.WriteAsync(_modelPath);
            return model;
        }

        public static void Evaluate(PredictionModel<WorldCupData, WorldCupPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<WorldCupData>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"RMS = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }
    }
}
