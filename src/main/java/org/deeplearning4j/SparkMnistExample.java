package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.ActivationMeanIterationListener;
import org.deeplearning4j.spark.impl.layer.SparkDl4jLayer;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import parquet.org.slf4j.Logger;
import parquet.org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author sonali
 */
public class SparkMnistExample {

    private static Logger log = LoggerFactory.getLogger(SparkMnistExample.class);

    public static void main(String[] args) throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf().set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, "true")
                .setMaster("local[*]")
                .setAppName("sparktest");


        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        log.info("Load data....");
        DataSetIterator iter = new MnistDataSetIterator(60000,60000);
        DataSet d = iter.next();
        d.shuffle();

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(784)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .seed(123)
                .constrainGradientToUnitNorm(true)
                .iterations(5).activationFunction("relu")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f)
                .momentum(0.5).constrainGradientToUnitNorm(true)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .hiddenLayerSizes(new int[]{600, 250, 200})
                .override(3, new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        List<DataSet> next = d.asList();


        JavaRDD<DataSet> data = sc.parallelize(next);



        MultiLayerNetwork network2 = master.fitDataSet(data);

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(), network2.output(d.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }
}
