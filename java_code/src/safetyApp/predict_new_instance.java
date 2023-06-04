package safetyApp;

import weka.classifiers.Classifier;
import weka.core.*;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

public class predict_new_instance {

    public static void practice_predict(String modelName) {
        String basePath = "./src/safetyApp/data/safety_recognition/";
        String modelBasePath = "src/safetyApp/model/";
        try {

            // Deserialize the classifier from a file
            Classifier clf = (Classifier) SerializationHelper.read(modelBasePath+modelName);
            System.out.println("Classifier deserialized successfully.");

            Random rand = new Random();

            ArrayList<Attribute> attributes = new ArrayList<>();
            for (int i = 0; i < 120; i++) {
                Attribute attribute = new Attribute("att" + i);
                attributes.add(attribute);
            }
            Instance instance = new DenseInstance(attributes.size());

            for (int i = 0; i < attributes.size(); i++) {
                instance.setValue(i, rand.nextFloat());
            }

            Instances trainingDataset = new Instances(new FileReader(basePath+"safety_recognition.arff"));
            trainingDataset.setClassIndex(trainingDataset.numAttributes() - 1);


            // Set the dataset for the instance
            instance.setDataset(trainingDataset);

            double prediction = clf.classifyInstance(instance);

            // Get the predicted class label
            String predictedClassLabel = instance.classAttribute().value((int) prediction);

            System.out.println("Predicted class: " + predictedClassLabel);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {

        practice_predict("tsf.model");
    }
}
