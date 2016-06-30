using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Sources;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(2));
            network.AddLayer(new BasicLayer(3));
            network.AddLayer(new BasicLayer(1));
            network.Structure.FinalizeStructure();
            network.Reset();

            var trainingDataSource = new CSVDataSource(@"Data\training.csv", true, ',');
            //var validationDataSource = new CSVDataSource(@"Data\validation.csv", true, ',');

            var trainingSet = new VersatileMLDataSet(trainingDataSource);
            //var validationSet = new VersatileMLDataSet(validationDataSource);

            trainingSet.Analyze();
            trainingSet.Normalize();

            var training = new ResilientPropagation(network, trainingSet);

            int epoch = 1;

            do
            {
                training.Iteration();
                Console.WriteLine($"Epoch #{epoch}. Error: {training.Error}");                epoch++;
            }
            while (training.Error > 0.01);

            training.FinishTraining();

            Console.WriteLine("Neural Network Results:");
            
            foreach(var pair in trainingSet)
            {
                var output = network.Compute(pair.Input);
                Console.WriteLine($"{pair.Input[0]},{pair.Input[1]}, actual={output[0]}, ideal={pair.Ideal}");
            }

            EncogFramework.Instance.Shutdown();
        }
    }
}
