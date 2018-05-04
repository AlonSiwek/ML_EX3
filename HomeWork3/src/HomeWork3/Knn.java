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

    Knn (int k, boolean weighting, DistanceCalculator distanceCalculator){
        this.k = k;
        this.weighting = weighting;
        this.distanceCalculator = distanceCalculator;
    }



    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
       data = instances;
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
     * Calculates the cross validation error, the average error on all folds.
     * @param instances Instances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds)
    {
        double crossValidationError = 0;
        Instances currTrainingData;
        Instances currValidationData;

        // shuffle the given data
        instances.randomize(new Random());


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
            
            data = currTrainingData;
            crossValidationError += calcAvgError(currValidationData);
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
        double weight = 0;

        while (!queue.isEmpty()) {
            distantInstance = queue.poll();
            instance = distantInstance.instance;
            distance = distantInstance.distance;

            //avoid zero division
            weight = (Math.pow(distance, 2) == 0) ? 0 : Math.pow(distance, -2);
            sumOfWeightedValues += weight * instance.classValue();
            sumOfWeights += weight;  
        }
        
        return (sumOfWeightedValues / sumOfWeights);
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