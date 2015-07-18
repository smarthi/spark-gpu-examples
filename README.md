# spark-gpu-examples
#Building and Running

Assuming [maven's mvn command](http://maven.apache.org) and git is on your path
Also assuming you [built nd4j](http://nd4j.org/getstarted.html)


1. git clone current master of deeplearning4j:
     
        git clone https://github.com/deeplearning4j/deeplearning4j
        git clone https://github.com/deeplearning4j/nd4j
        git clone https://github.com/deeplearning4j/Canova
        cd nd4j
        mvn clean install -DskipTests -Dmaven.javadoc.skip=true
        cd Canova
        mvn clean install -DskipTests -Dmaven.javadoc.skip=true
        cd deeplearning4j
        mvn clean install -DskipTests -Dmaven.javadoc.skip=true

After wards build the spark examples:

    git clone https://github.com/deeplearning4j/spark-gpu-examples
    cd spark-examples
    mvn clean install -DskipTests

After you're comfortable, you can change backends in the pom and re run :

./runandtrain.sh





    

Run the examples in src/main/java with the uber jar:
java -cp target/spark-examples-*.jar 
