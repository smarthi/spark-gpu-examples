package org.deeplearning4j;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.StandardScaler;

import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.LibSvmRecordWriter;
import org.canova.api.records.writer.impl.LineRecordWriter;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.layer.SparkDl4jLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.File;

/**
 * @author Adam Gibson
 */
public class LogisticRegressionComparison {

    public static void main(String[] args) throws Exception {
        SparkConf conf = new SparkConf().setAppName("Logistic Classifier Example").setMaster("local[*]");
        SparkContext sc = new SparkContext(conf);
        RecordReader svmLightReader = new SVMLightRecordReader();
        String path = "src/main/resources/data/svmLight/iris_svmLight_0.txt";
        String outPath = "iris_svmlight_out.txt";
        svmLightReader.initialize(new FileSplit(new File(path)));
        Configuration writeConf = new Configuration();
        writeConf.set(LineRecordWriter.PATH, outPath);


        RecordWriter writer = new LibSvmRecordWriter();
        writer.setConf(writeConf);
        while(svmLightReader.hasNext())
            writer.write(svmLightReader.next());

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, outPath).toJavaRDD().map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                return new LabeledPoint(v1.label(), Vectors.dense(v1.features().toArray()));
            }
        }).cache();


        StandardScaler scaler = new StandardScaler(true,true);

        final StandardScalerModel model2 = scaler.fit(data.map(new Function<LabeledPoint, Vector>() {

            @Override
            public Vector call(LabeledPoint v1) throws Exception {
                return v1.features();
            }
        }).rdd());

        JavaRDD<LabeledPoint> normalizedData = data.map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                Vector features = v1.features();
                Vector normalized = model2.transform(features);
                return new LabeledPoint(v1.label(), normalized);
            }
        }).cache();

        // Split initial RDD into two... [60% training data, 40% testing data].
        JavaRDD<LabeledPoint>[] splits = normalizedData.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];
        StopWatch watch = new StopWatch();
        // Run training algorithm to build the model.
        LogisticRegressionWithLBFGS model = new LogisticRegressionWithLBFGS().setNumClasses(3);
        model.optimizer().setMaxNumIterations(10);
        long start = System.currentTimeMillis();
        final LogisticRegressionModel model3 = model.run(training.rdd());
        long end = System.currentTimeMillis();
        System.out.println("Time for spark " + Math.abs(end - start));
        watch.reset();
        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model3.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );




        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.fMeasure();
        System.out.println("F1 = " + precision);

        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).
                        optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(10).weightInit(WeightInit.XAVIER)
                .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();


        System.out.println("Initializing network");
        final SparkDl4jLayer master = new SparkDl4jLayer(sc,neuralNetConfiguration);
        start =System.currentTimeMillis();
        master.fit(new JavaSparkContext(sc), splits[0]);
        end =System.currentTimeMillis();
        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabelsDl4j = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        double prediction = MLLibUtil.toClassifierPrediction(master.predict(p.features()));
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        MulticlassMetrics dl4jMetrics = new MulticlassMetrics(predictionAndLabelsDl4j.rdd());
        double dl4jPrecision = dl4jMetrics.fMeasure();
        System.out.println("F1 = " + dl4jPrecision);
        System.out.println("Time for dl4j " + Math.abs(end - start));



    }

}
