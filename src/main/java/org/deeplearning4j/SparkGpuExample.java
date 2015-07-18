package org.deeplearning4j;

import java.io.File;
import java.security.AccessController;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import akka.actor.ActorSystem;
import akka.actor.ExtendedActorSystem;
import akka.serialization.JavaSerializer;
import akka.serialization.Serialization;
import akka.serialization.SerializationExtension;
import akka.serialization.Serializer;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.serializer.KryoRegistrator;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.serializer.SerializationDebugger;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.layer.SparkDl4jLayer;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.SerializationTester;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

/**
 * @author sonali
 */
public class SparkGpuExample {

    public static void main(String[] args) throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf().set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION,"true")
                .setMaster("local[*]")
                .setAppName("sparktest");


        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(10).weightInit(WeightInit.XAVIER)
                .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();



        System.out.println("Initializing network");
        SparkDl4jLayer master = new SparkDl4jLayer(sc,conf);
        DataSet d = new IrisDataSetIterator(150,150).next();
        d.normalizeZeroMeanZeroUnitVariance();
        d.shuffle();
        List<DataSet> next = d.asList();


        JavaRDD<DataSet> data = sc.parallelize(next);



        OutputLayer network2 =(OutputLayer) master.fitDataSet(data);

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }
}
