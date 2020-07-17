using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace MyFirstMLApp
{

    public class FeedBackTrainingData
    {
        [Column(ordinal: "0", name: "Label")]
        public bool isGood { get; set; }

        [Column(ordinal: "1")]
        public string FeedBackText { get; set; }


    }

    public class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]

        public bool isGood { get; set; }
        class Program
        {
            static List<FeedBackTrainingData> trainingdata = new List<FeedBackTrainingData>();
            static List<FeedBackTrainingData> testdata = new List<FeedBackTrainingData>();
            static void LoadTraningData()
            {
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "This is good",
                    isGood = true
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "This is bad",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "shit",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "This is horrible",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "This is worst",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "sweet and nice",
                    isGood = true
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "nice and good",
                    isGood = true
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "very good",
                    isGood = true
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "very very bad",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "quit average",
                    isGood = false
                });
                trainingdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "bad and hell",
                    isGood = false
                });

            }
            static void LoadTestData()
            {
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "good",
                    isGood = true
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "bad",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "hell",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "horrible",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "okay",
                    isGood = true
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "shit",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "sweet",
                    isGood = true
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "worst",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "nice",
                    isGood = true
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "nice",
                    isGood = true
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "average",
                    isGood = false
                });
                testdata.Add(new FeedBackTrainingData()
                {
                    FeedBackText = "quit",
                    isGood = false
                });

            }
            static void Main(string[] args)
            {
                LoadTraningData();

                var mlContext = new MLContext();

                IDataView dataview = mlContext.CreateStreamingDataView<FeedBackTrainingData>(trainingdata);

                var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features").Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

                var model = pipeline.Fit(dataview);

                LoadTestData();

                IDataView dataview1 = mlContext.CreateStreamingDataView<FeedBackTrainingData>(testdata);

                var predictions = model.Transform(dataview1);

                var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

                Console.WriteLine(metrics.Accuracy);
                //Console.ReadKey();

                string str = "Y";


                while(str == "Y")
                {
                    Console.WriteLine("Enter Feedback please : ");
                    string feedbackstr = Console.ReadLine().ToString();

                    var predictionFucntion = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);

                    var feedbackinput = new FeedBackTrainingData();

                    feedbackinput.FeedBackText = feedbackstr;

                    var feedbackpredicted = predictionFucntion.Predict(feedbackinput);

                    Console.WriteLine("Predicted:--" + feedbackpredicted.isGood);
                }
               
                Console.ReadLine();

            }
        }
    }
}
