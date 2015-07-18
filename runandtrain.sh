#!/usr/bin/env bash

mvn clean install -DskipTests
java -Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=5005,suspend=n -Xmx5g -Xms5g -cp target/spark-gpu-examples-1.0-SNAPSHOT.jar org.deeplearning4j.LogisticRegressionComparison.java
