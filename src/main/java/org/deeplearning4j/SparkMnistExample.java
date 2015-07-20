package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import parquet.org.slf4j.Logger;
import parquet.org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Collections;
import java.util.List;

/**
 * @author sonali
 */
public class SparkMnistExample {

    private static Logger log = LoggerFactory.getLogger(SparkMnistExample.class);

    public static void main(String[] args) throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf().set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, "false")
                .setMaster("local[*]")
                .setAppName("sparktest");


        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        log.info("Load data....");


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(784)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .seed(123)
                .constrainGradientToUnitNorm(true)
                .iterations(5).activationFunction("softplus")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).batchSize(1000)
                .momentum(0.5).constrainGradientToUnitNorm(true)
                .momentumAfter(Collections.singletonMap(3, 0.9)).l1(0.3).l2(1e-3).regularization(true)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(4).backward(true)
                .hiddenLayerSizes(600, 250, 200)
                .override(3, new ClassifierOverride())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        //number of partitions should be partitioned by batch size
        JavaRDD<String> lines = sc.textFile("s3n://dl4j-distribution/mnist_svmlight.txt",conf.getConf(0).getBatchSize() / 60000);
        RecordReader svmLight = new SVMLightRecordReader();
        Configuration canovaConf = new Configuration();
        //number of features + label
        canovaConf.setInt(SVMLightRecordReader.NUM_ATTRIBUTES,784);
        svmLight.setConf(canovaConf);

        JavaRDD<DataSet> data = lines.map(new RecordReaderFunction(svmLight, 784, 10));
        MultiLayerNetwork network2 = master.fitDataSet(data);
        FileOutputStream fos  = new FileOutputStream("params.txt");
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(dos,network2.params());
        dos.flush();
        dos.close();


        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter = new MnistDataSetIterator(1000,60000);
        Evaluation eval = new Evaluation(10);
        while(iter.hasNext()) {
            DataSet next = iter.next();
            eval.eval(next.getLabels(),model.output(next.getFeatureMatrix(), true));
        }

        System.out.println(eval.stats());


    }
}
