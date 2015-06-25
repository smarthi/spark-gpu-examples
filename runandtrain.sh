#!/usr/bin/env bash

mvn clean install -DskipTests
java -Xmx5g -Xms5g -cp target/spark-gpu-examples-1.0-SNAPSHOT.jar org.deeplearning4j.SparkGpuExample
