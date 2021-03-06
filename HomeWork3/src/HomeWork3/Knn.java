package HomeWork3;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import java.util.Comparator;

class DistanceComparator implements Comparator<DistantInstance> {

    @Override
    public int compare(DistantInstance x, DistantInstance y)
    {
        return (x.distance > y.distance) ? -1 : 1;
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
    boolean efficiency;

    public DistanceCalculator(int p, boolean efficiency){
        this.p = p;
        this.efficiency = efficiency;
    }

    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it as a class variables.
     */
    public double distance (Instance one, Instance two) {

        if (p == Integer.MAX_VALUE) {
            if(efficiency){
               return efficientLInfinityDistance(one, two);
            }
            return lInfinityDistance(one, two);
        }

        if(efficiency){
            return efficientLpDistance(one, two);
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
        double currDistance;
        double maxDistance = 0;

        //check the max distance
        for (int i = 0; i < one.numAttributes() - 1; i++){
            currDistance = Math.abs((one.value(i) - two.value(i)));
            if (currDistance > maxDistance){
                maxDistance = currDistance;
            }
        }
        return maxDistance;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two) {
        double distance = 0;

        for (int i = 0; i < one.numAttributes() - 1; i++) {
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
            if(distance > Math.pow(thresholdDistance, p)){
                break;
            }
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
    private double efficientLInfinityDistance(Instance one, Instance two) {
        double currDistance;
        double maxDistance = 0;

        //check the max distance
        for (int i = 0; i < one.numAttributes() - 1; i++) {
            currDistance = Math.abs((one.value(i) - two.value(i)));
            if (currDistance > maxDistance) {
                maxDistance = currDistance;
            }

            if (maxDistance > thresholdDistance) {
                break;
            }
        }
        return maxDistance;
    }

    public void resetThresholdDistance(){
        thresholdDistance = Double.MAX_VALUE;
    }
}

public class Knn implements Classifier {
    private Instances data;
    private DistanceCalculator distanceCalculator;
    private boolean weighting;
    private int k;

    // class constructor
    Knn (int k, boolean weighting, DistanceCalculator distanceCalculator){
        this.k = k;
        this.weighting = weighting;
        this.distanceCalculator = distanceCalculator;
    }



    @Override
    /** mandatory override method that is not used
     */
    public void buildClassifier(Instances instances) throws Exception {
    }


    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        PriorityQueue<DistantInstance> queue = findNearestNeighbors(instance);

        if (weighting) {
            return getWeightedAverageValue(queue);
        }
        return getAverageValue(queue);
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
     * separates data into all separations to training and validation data for a given number of folds
     * @param instances data to split
     * @param num_of_folds The number of folds to use.
     * @return a 2D-array of where each element represents a separation where the first (inner) element is training data and the second validation data
     */
    public Instances[][] separateData(Instances instances, int num_of_folds){
        Instances[][] res = new Instances[num_of_folds][2];
        Instances currTrainingData;
        Instances currValidationData;

        for (int i = 0; i < num_of_folds; i++) {
            currTrainingData = new Instances(instances,0);
            currValidationData = new Instances(instances,0);

            // separate data into training and validation
            for (int j = 0; j < instances.size(); j++) {
                if(j % num_of_folds == i){
                    currValidationData.add(instances.instance(j));
                } else {
                    currTrainingData.add(instances.instance(j));
                }
            }
            res[i][0] = currTrainingData;
            res[i][1] = currValidationData;
        }
        return res;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param instances 2D-array of Instances the size of num_of_folds*2 where each element represents
     *                 a separation where the first (inner) element is training data and the second validation data
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances[][] instances, int num_of_folds)
    {
        double crossValidationError = 0;
        for (int i = 0; i < num_of_folds; i++) {
            data = instances[i][0];
            crossValidationError += calcAvgError(instances[i][1]);
        }
        return crossValidationError / num_of_folds;
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public PriorityQueue<DistantInstance> findNearestNeighbors(Instance instance) {
        Comparator<DistantInstance> comparator = new DistanceComparator();
        PriorityQueue<DistantInstance> queue = new PriorityQueue<>(k, comparator);
        DistantInstance currDistantInstance;
        double currDistance;
        double threshold;
        int i = 0;

        distanceCalculator.resetThresholdDistance();
        for (Instance currInstance : data) {
            currDistance = distanceCalculator.distance(instance, currInstance);
            currDistantInstance = new DistantInstance(currDistance, currInstance);

            if(i < k){
                queue.add(currDistantInstance);
            } else {
                threshold = queue.peek().distance;
                distanceCalculator.thresholdDistance = threshold;
                if(currDistance < threshold){
                    queue.remove();
                    queue.add(currDistantInstance);
                }
            }

            i++;
        }
        return queue;
    }

    /**
     * Calculates the average value of the given elements in the collection.
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
        double sumOfWeightedValues = 0;
        double sumOfWeights = 0;
        double weight;

        while (!queue.isEmpty()) {
            distantInstance = queue.poll();
            instance = distantInstance.instance;
            distance = distantInstance.distance;

            //avoid zero division
            weight = (Math.pow(distance, 2) == 0) ? 0 : Math.pow(distance, -2);
            sumOfWeightedValues += weight * instance.classValue();
            sumOfWeights += weight;  
        }

        return (sumOfWeights == 0) ? 0 : sumOfWeightedValues / sumOfWeights;
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