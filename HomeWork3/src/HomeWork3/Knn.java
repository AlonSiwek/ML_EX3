package HomeWork3;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.lang.reflect.Array;
import java.util.*;

import java.util.Comparator;

class DistanceComperator implements Comparator<DistantInstance> {

    @Override
    public int compare(DistantInstance x, DistantInstance y)
    {
        return (x.distance > y.distance) ? 1 : -1;
    }
}

class DistantInstance{
    double distance;
    Instance instance;

    public DistantInstance(double distance,Instance instance){
        this.distance = distance;
        this.instance = instance;
    }
}
class DistanceCalculator {
    int p;
    double thresholdDistance;
    boolean efficiency = false;


    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it as a class variables.
     */
    public double distance (Instance one, Instance two) {

        //check whether we calculate efficiency or not and calculate accordindgly
        if (p == Integer.MAX_VALUE) {
            return lInfinityDistance(one, two);
        }

        return lpDistance(one, two);
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two) {
         double distance = 0;

        //sumerize the distances of the dimension of the vector
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
        }

        distance = Math.pow(distance, (1.0 / p));

        return distance;
    }

    /**
     * Returns the Lp infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double distance = 0;

        //check the max distance
        for (int i = 0; i < one.numAttributes() - 1; i++){
            if (Math.abs((one.value(i) - two.value(i))) > 0)
                distance = Math.abs((one.value(i) - two.value(i)));
        }
        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDisatnce(Instance one, Instance two, int p, double neighborDistance) {
        double distance = 0;

        //sumerize the distances of the dimension of the vector
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
        }

        distance = Math.pow(distance, (1.0 / p));

        return distance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double neighborDistance) {
        double distance = 0;

        for (int i = 0; i < one.numAttributes() - 1; i++){
            if (Math.abs((one.value(i) - two.value(i))) > 0)
                distance = Math.abs((one.value(i) - two.value(i)));
            if (distance > neighborDistance)
                break;
        }

        return distance;
    }
}

public class Knn implements Classifier {
    private Instances data;
    private boolean weighting;
    private int k;
    private int p;

    Knn (int k, int p, boolean weighting){
        this.k = k;
        this.p = p;
        this.weighting = weighting;
    }



    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
//        data = instances;
    }


    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        PriorityQueue<DistantInstance> queue = findNearestNeighbors(instance);

        if (weighting) {
            return getAverageValue(queue);
        }

        return getWeightedAverageValue(queue);
    }

    /**
     * Calculates the average error on a given set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     * @param instances
     * @return
     */
    public double calcAvgError (Instances instances){
        double totalError = 0;

        for (Instance instance: instances) {
            totalError += Math.abs(instance.classValue() - regressionPrediction(instance));
        }

        return totalError / (double) instances.size();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param instances Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds)
    {
        double crossValidationError = 0;
        Instances [] instancesArray = new Instances[num_of_folds];

        // shuffle the given data
        instances.randomize(new Random());

        // Initialize the instances folds
        for (int i = 0; i < instancesArray.length; i++) {
            instancesArray[i] = new Instances(instances, instances.size());
        }

        // Fill the folds with the given instances
        for (int i = 0; i < instances.size(); i++) {
            instancesArray[i % num_of_folds].add(instances.instance(i));
        }

        // Loop over all folds and calculate the avg cross validation error
        // Each time validate with a different fold
        for (int i = 0; i < num_of_folds; i++) {
            Instances currTrainingData = new Instances(instances, 0);
            for (int j = 0; j < num_of_folds; j++){
                 if (j != i){
                     for (Instance instance : instancesArray[j]) {
                         currTrainingData.add(instance);
                     }
                 }
            }
            // Set training data
            data = currTrainingData;
            // Validate using the left out validation fold
            crossValidationError += calcAvgError(instancesArray[i]);
        }

        return crossValidationError / num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public PriorityQueue<DistantInstance> findNearestNeighbors(Instance instance) {
        Comparator<DistantInstance> comparator = new DistanceComperator();
        PriorityQueue<DistantInstance> queue = new PriorityQueue<>(0, comparator);
        DistanceCalculator distanceCalculator = new DistanceCalculator();
        DistantInstance currDistantInstance;
        double currDistance;
        double threshold;

        for (Instance currInstance : data) {
            currDistance = distanceCalculator.distance(instance, currInstance);
            currDistantInstance = new DistantInstance(currDistance, currInstance);

            if(queue.size() < k){
                queue.add(currDistantInstance);
            }

            if(queue.size() == k) {
                threshold = queue.peek().distance;
                distanceCalculator.thresholdDistance = threshold;
                if(currDistance < threshold){
                    queue.remove();
                    queue.add(currDistantInstance);
                }
            }
        }
        return queue;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (PriorityQueue<DistantInstance> queue) {
        double sum = 0;

        while (!queue.isEmpty()) {
            sum += queue.poll().instance.classValue();
        }

        return (sum / k);
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(PriorityQueue<DistantInstance> queue) {
        DistantInstance distantInstance;
        double distance;
        Instance instance;
        double numerator = 0;
        double denominator = 0;
        double wi;

        //TODO verify correctness
        while (!queue.isEmpty()) {
            distantInstance = queue.poll();
            instance = distantInstance.instance;
            distance = distantInstance.distance;

            if (distance == 0) {
                return instance.classValue();
            }

            wi = 1.0 / (Math.pow(distance, 2));
            if (wi > 0) {
                numerator += wi * instance.value(data.classIndex());
                denominator += wi;
            }
        }


        return (numerator/denominator);
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}