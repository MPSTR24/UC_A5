package safetyApp;
import tsml.classifiers.interval_based.TSF;
import tsml.classifiers.kernel_based.ROCKETClassifier;
import tsml.classifiers.legacy.elastic_ensemble.DTW1NN;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import java.io.*;
import java.util.Arrays;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws Exception {


        RandomForest rf = new RandomForest();

        TSF tsf = new TSF();

        DTW1NN knn = new DTW1NN();

        ROCKETClassifier rocket = new ROCKETClassifier();
        rocket.setNumKernels(6250);
        rocket.setNormalise(true);
        rocket.enableMultiThreading(Runtime.getRuntime().availableProcessors());

//        ClassifierUtils.runEvaluation("safety_recognition.arff", rf, "RF_results.txt");
//        ClassifierUtils.runEvaluation("safety_recognition.arff", tsf, "TSF_results.txt");
//        ClassifierUtils.runEvaluation("safety_recognition.arff", knn, "DTW_results.txt");
//        ClassifierUtils.runEvaluation("safety_recognition.arff", rocket, "ROCKET_results.txt");
//
//        ClassifierUtils.serialiseModel(rocket, "safety_recognition.arff", "rocket.model");
//        ClassifierUtils.serialiseModel(tsf, "safety_recognition.arff", "tsf.model");


//        ClassifierUtils.deserialiseModel("tsf.model");

    }
}
