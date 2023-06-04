package safetyApp;

import tsml.classifiers.kernel_based.ROCKETClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class Pair {
    int numKernels;
    double accuracy;

    Pair(int numKernels, double accuracy) {
        this.numKernels = numKernels;
        this.accuracy = accuracy;
    }
}

public class RocketTesting {

    public static void main(String[] args) throws Exception {

        System.out.println("Starting rocket test");

        ArffLoader loader = new ArffLoader();
        File file = new File("./src/safetyApp/data/safety_recognition/safety_recognition.arff");
        loader.setFile(file);
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        int lowerBound = 5500;
        int upperBound = 7000;
        int interval = 1000;

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        System.out.println("Spawning "+Runtime.getRuntime().availableProcessors()+" executors");

        double prevBestAccuracy = 0;
        while (upperBound - lowerBound > interval) { // Stop refining when the range is small enough
            int midPoint = (upperBound + lowerBound) / 2;

            // Randomly choose a bound to evaluate
            int boundToEvaluate = (new Random().nextBoolean()) ? lowerBound : upperBound;

            Future<Pair> boundFuture = executor.submit(() -> getAccuracyPair(boundToEvaluate, data));
            Future<Pair> midFuture = executor.submit(() -> getAccuracyPair(midPoint, data));

            Pair boundPair = boundFuture.get();
            Pair midPair = midFuture.get();

            // Print the results
            System.out.println("Kernels: " + boundPair.numKernels + " | Accuracy: " + boundPair.accuracy);
            System.out.println("Kernels: " + midPair.numKernels + " | Accuracy: " + midPair.accuracy);

            // Narrow the range based on the best accuracy
            if (midPair.accuracy > boundPair.accuracy) {
                if (boundToEvaluate == lowerBound) {
                    lowerBound = midPoint;
                } else {
                    upperBound = midPoint;
                }
            } else {
                if (boundToEvaluate == lowerBound) {
                    upperBound = midPoint;
                } else {
                    lowerBound = midPoint;
                }
            }

            // If the best accuracy isn't improving much, break the loop
            double bestAccuracy = Math.max(midPair.accuracy, boundPair.accuracy);
            if (bestAccuracy - prevBestAccuracy < 0.01) {
                break;
            }
            prevBestAccuracy = bestAccuracy;
        }



        // Final round to get the best numKernels and accuracy
        int finalLowerBound = lowerBound;
        int finalUpperBound = upperBound;
        Future<Pair> bestFuture = executor.submit(() -> {
            int bestNumKernels = finalLowerBound;
            double bestAccuracy = getAccuracyPair(finalLowerBound, data).accuracy;

            for (int numKernels = finalLowerBound + 1; numKernels <= finalUpperBound; numKernels++) {
                double accuracy = getAccuracyPair(numKernels, data).accuracy;
                if (accuracy > bestAccuracy) {
                    bestAccuracy = accuracy;
                    bestNumKernels = numKernels;
                }
            }

            return new Pair(bestNumKernels, bestAccuracy);
        });

        Pair bestPair = bestFuture.get();

        executor.shutdown();

        System.out.println("\nBest kernels: " + bestPair.numKernels + " with accuracy: " + bestPair.accuracy);

    }

    private static Pair getAccuracyPair(int numKernels, Instances data) throws Exception {
        ROCKETClassifier tempRocket = new ROCKETClassifier();
        tempRocket.setNumKernels(numKernels);
        tempRocket.buildClassifier(data);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tempRocket, data, 10, new Random(1));
        double accuracy = eval.pctCorrect();
        return new Pair(numKernels, accuracy);
    }
}
