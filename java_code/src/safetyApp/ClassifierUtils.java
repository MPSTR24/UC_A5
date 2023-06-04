package safetyApp;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;


/**
 * Separate class for the classifier utils allows for separate instances to be made for threads, avoiding a shared
 * mutable state.
 */
public class ClassifierUtils {

    private static final String BASE_PATH = "./src/safetyApp/data/safety_recognition/";
    private static final String RESULTS_BASE_PATH = "src/safetyApp/results/";
    private static final String MODEL_BASE_PATH = "src/safetyApp/model/";

    public static Instances loadDataSet(String dataPath) throws Exception{
        try (FileReader reader = new FileReader(BASE_PATH + dataPath)) {
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (IOException e) {
            System.err.println("Failed to load data set. Error: " + e.getMessage());
            throw e;
        }
    }

    public static void runEvaluation(String dataPath, Classifier clf, String resultsPath) throws Exception {

        // Load the dataset
        Instances data = loadDataSet(dataPath);

        // Perform cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(clf, data, 10, data.getRandomNumberGenerator(44));

        // Get the evaluation summary
        String summary = eval.toSummaryString();

        // Write the summary to a text file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(RESULTS_BASE_PATH + resultsPath))) {
            writer.write(summary);
        }

        System.out.println("Cross-validation completed successfully.");

    }
    public static void serialiseModel(Classifier clf, String dataPath, String modelName) throws Exception {

        // Load the dataset
        Instances data = loadDataSet(dataPath);

        clf.buildClassifier(data);

        // Serialise the classifier to a file
        SerializationHelper.write((MODEL_BASE_PATH + modelName), clf);

        System.out.println("Classifier serialized successfully.");

    }

    public static void deserialiseModel(String modelName) {

        try {

            // Deserialize the classifier from a file
            Classifier clf = (Classifier) SerializationHelper.read(MODEL_BASE_PATH+modelName);
            System.out.println("Classifier deserialized successfully.");


            // Load the test instances
            try (FileReader reader = new FileReader(BASE_PATH+"safety_recognition.arff")) {
                Instances testInstances = new Instances(reader);
                testInstances.setClassIndex(testInstances.numAttributes() - 1);

                // Make predictions on test instances
                for (int i = 0; i < testInstances.numInstances(); i++) {
                    Instance instance = testInstances.instance(i);
                    double prediction = clf.classifyInstance(instance);
                    System.out.println("Instance " + i + ": Predicted class = " + testInstances.classAttribute().value((int) prediction));
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
