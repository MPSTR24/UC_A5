package safetyApp;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.kernel_based.ROCKETClassifier;
import tsml.classifiers.legacy.elastic_ensemble.DTW1NN;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class cv_fold {

    public static void runEvaluation(String dataPath, Classifier clf, String resultsPath) {
        String basePath = "./src/main/java/safetyApp/data/safety_recognition/";
        String resultsBasePath = "src/main/java/safetyApp/results/";


        try {

            // Load the dataset
            Instances data = new Instances(new FileReader(basePath+dataPath));
            data.setClassIndex(data.numAttributes() - 1);

            // Perform cross-validation
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(clf, data, 10, data.getRandomNumberGenerator(44));

            // Get the evaluation summary
            String summary = eval.toSummaryString();

            // Write the summary to a text file
            BufferedWriter writer = new BufferedWriter(new FileWriter(resultsBasePath + resultsPath));
            writer.write(summary);
            writer.close();

            System.out.println("Cross-validation completed successfully.");

        } catch (Exception e) {
        e.printStackTrace();
    }

    }
    public static void serialiseModel(Classifier clf, String dataPath, String modelName) {

        try {
            String basePath = "./src/main/java/safetyApp/data/safety_recognition/";
            String modelBasePath = "src/main/java/safetyApp/model/";

            // Load the dataset
            Instances data = new Instances(new FileReader(basePath+dataPath));
            data.setClassIndex(data.numAttributes() - 1);


            clf.buildClassifier(data);

            // Serialise the classifier to a file
            SerializationHelper.write((modelBasePath + modelName), clf);

            System.out.println("Classifier serialized successfully.");


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void deserialiseModel(String modelName) {
        String basePath = "./src/main/java/safetyApp/data/safety_recognition/";
        String modelBasePath = "src/main/java/safetyApp/model/";
        try {

            // Deserialize the classifier from a file
            Classifier clf = (Classifier) SerializationHelper.read(modelBasePath+modelName);
            System.out.println("Classifier deserialized successfully.");


            // Load the test instances
            Instances testInstances = new Instances(new FileReader(basePath+"safety_recognition.arff"));
            testInstances.setClassIndex(testInstances.numAttributes() - 1);

            // Make predictions on test instances
            for (int i = 0; i < testInstances.numInstances(); i++) {
                Instance instance = testInstances.instance(i);
                double prediction = clf.classifyInstance(instance);
                System.out.println("Instance " + i + ": Predicted class = " + testInstances.classAttribute().value((int) prediction));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {


        RandomForest rf = new RandomForest();

        TSF tsf = new TSF();

        DTW1NN knn = new DTW1NN();

        ROCKETClassifier rocket = new ROCKETClassifier();
        rocket.setNumKernels(1000);


//        runEvaluation("safety_recognition.arff", rf, "RF_results.txt");
//        runEvaluation("safety_recognition.arff", tsf, "TSF_results.txt");
//        runEvaluation("safety_recognition.arff", knn, "DTW_results.txt");
//        runEvaluation("safety_recognition.arff", rocket, "ROCKET_results.txt");
//
//        serialiseModel(rocket, "safety_recognition.arff", "rocket.model");
//        serialiseModel(tsf, "safety_recognition.arff", "tsf.model");

        deserialiseModel("tsf.model");


    }
}
