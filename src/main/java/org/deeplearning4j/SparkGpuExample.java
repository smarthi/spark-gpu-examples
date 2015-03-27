package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;

import java.util.Map;

/**
 * @author sonali
 */
public class SparkGpuExample {

    public static void main(String[] args) throws Exception {

        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]").set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION,"false")
                .set("spark.akka.frameSize", "100")
                .setAppName("mnist");

        System.out.println("Setting up Spark Context...");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        Map<Integer,OutputPreProcessor> preProcessorMap = new HashMap<>();
        for(int i = 0; i < 3; i++)
            preProcessorMap.put(i,new BinomialSamplingPreProcessor());

        int batchSize = 5000;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9).iterations(1)
                .constrainGradientToUnitNorm(true).weightInit(WeightInit.DISTRIBUTION)
                .dist(Distributions.normal(new MersenneTwister(123), 1e-4))
                .nIn(784).nOut(10).layerFactory(LayerFactories.getFactory(RBM.class))
                .list(4).hiddenLayerSizes(600, 500, 400)
                .override(new ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {

                        if (i == 3) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();




        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        System.out.println("Initializing network");
        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);
        DataSet d = new MnistDataSetIterator(60000,60000).next();
        List<DataSet> next = new ArrayList<>();
        for(int i = 0; i < d.numExamples(); i++)
            next.add(d.get(i).copy());

        JavaRDD<DataSet> data = sc.parallelize(next);
        MultiLayerNetwork network2 = master.fitDataSet(data);

        Evaluation evaluation = new Evaluation();
        evaluation.eval(d.getLabels(),network2.output(d.getFeatureMatrix()));
        System.out.println("Averaged once " + evaluation.stats());


        INDArray params = network2.params();
        Nd4j.writeTxt(params,"params.txt",",");
        FileUtils.writeStringToFile(new File("conf.json"),network2.getLayerWiseConfigurations().toJson());
    }
}
