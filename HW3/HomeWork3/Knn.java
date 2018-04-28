package HomeWork3;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import java.util.*;

class DistanceCalculator {

    public boolean m_Efficient = false;
    /**
     * We leave it up to you whether you want the distance method to get all relevant
     * parameters(lp, efficient, etc..) or have it as a class variables.
     */
    public double distance (Instance one, Instance two, int p, double neighborDistance) {
        //check wheater we calculate efficiency or not and calculate accordindgly
        if (m_Efficient == false){
            if (p > 3)
                return lInfinityDistance(one,two);
            else
                return lpDistance(one, two, p);
        }else{
            if (p > 3)
                return efficientLInfinityDistance(one,two, neighborDistance);
            else
                return efficientLpDisatnce(one,two, p, neighborDistance);
        }

    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, int p) {
         double distance = 0;
        //sumerize the distances of the dimension of the vector
        for (int i = 0; i < one.numAttributes() - 1; i++)
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);

        if (p <= 3){
            if (p == 2)
                distance = Math.sqrt(distance);
            if (p == 3)
                distance = Math.cbrt(distance);
        }

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
        //chack the max distance of the vector dimension
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

        for (int i = 0; i < one.numAttributes() - 1; i++){
            distance += Math.pow(Math.abs(one.value(i) - two.value(i)), p);
            if (distance > Math.pow(neighborDistance, p))
                break;
        }

        if (p <= 3){
            if (p == 2)
                distance = Math.sqrt(distance);
            if (p == 3)
                distance = Math.cbrt(distance);
        }

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
            if (Math.abs((one.value(i)-two.value(i))) > 0)
                distance = Math.abs((one.value(i)-two.value(i)));
            if (distance > neighborDistance)
                break;
        }

        return distance;
    }
}

public class Knn implements Classifier {

    private double m_kNeighborDist = 0;
    protected boolean distEffCheck = false;
    private Instances m_trainingInstances;
    private int k; // {1,2,...,20}
    private int p; // {1,2,3,infinity} // p = 4 means infinity
    private int m_weightingScheme; // 0 for uniform , 1 for weighted

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_trainingInstances = instances;
    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        HashSet<Instance> hashSet = findNearestNeighbors(instance);
        Iterator<Instance> iterator = hashSet.iterator();
        if (m_weightingScheme == 0 || m_weightingScheme == 1){
            //uniform
            if (m_weightingScheme == 0)
                return getAverageValue(instance, iterator, hashSet.size());
            //weighted
            if (m_weightingScheme == 1)
                return getWeightedAverageValue(instance, iterator);
        }
        return -1;
    }

    /**
     * Calculates the average error on a given set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all instances.
     * @param instances
     * @return
     */
    public double calcAvgError (Instances instances){
        double err = 0, avgErr;

        for (int i = 0; i < instances.size(); i++)
            err += Math.abs((instances.instance(i).value(m_trainingInstances.classIndex())) -
                    (regressionPrediction(instances.instance(i))));

        avgErr = err / (double) instances.size();

        return avgErr;
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param arr Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances[] arr, int num_of_folds, int validationIndex){
        return calcAvgError(arr[validationIndex]);
    }


    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public HashSet<Instance> findNearestNeighbors(Instance instance) {
        DistanceCalculator distCalc = new DistanceCalculator();
        distCalc.m_Efficient = distEffCheck;
        // this map will hold all the instances as keys and their distances from instace as value
        Map<Instance, Double> map = new HashMap<>();
        HashSet<Instance> kNeighbors = new HashSet<>();


        return distEffCheck ? efficientNearestNeighbors(instance, map, kNeighbors, distCalc) :unEfficientFindNearestNeighbors(instance, kNeighbors, distCalc, map);
    }

    private HashSet<Instance> efficientNearestNeighbors(Instance instance, Map<Instance, Double> map, HashSet<Instance> kNeighbors, DistanceCalculator distCalc)
    {
        Instance secondInstance;
        double dist;
        double kNeighborDistance = -1;
        Instance kNeighbor = null;
        distCalc.m_Efficient = false;
        distEffCheck = false;
        InstanceComparator ic = new InstanceComparator();
        // at first get k neighbors with non-efficient calculation
        int i = 0;
        while (i < k || map.size() < k)
        {
            secondInstance = m_trainingInstances.instance(i++);
            dist = distCalc.distance(instance, secondInstance, p, kNeighborDistance);
            if (ic.compare(instance,secondInstance) != 0)
            {
                map.put(secondInstance, dist);
                if (dist > kNeighborDistance)
                {
                    kNeighborDistance = dist;
                    kNeighbor = secondInstance;
                }
            }
        }

        distEffCheck = true;
        distCalc.m_Efficient = true;
        // check efficiently the dist of rest of the instances.
        for (int j = i; j < m_trainingInstances.size(); j++)
        {
            secondInstance = m_trainingInstances.instance(j);
            dist = distCalc.distance(instance,secondInstance,p, kNeighborDistance);
            if (dist < kNeighborDistance)
            {
                map.remove(kNeighbor);
                map.put(secondInstance, dist);
                // find the new kNeighbor and his dist
                kNeighborDistance = -1;
                for (Instance key : map.keySet())
                {
                    dist = map.get(key);
                    if (dist > kNeighborDistance)
                    {
                        kNeighbor = key;
                        kNeighborDistance = dist;
                    }
                }
            }
        }

        // return the chosen k neighbors
        for (Instance key : map.keySet())
        {
            kNeighbors.add(key);
        }

        m_kNeighborDist = kNeighborDistance; // set the kNeighborDist for other methods
        return kNeighbors;
    }

    private HashSet<Instance> unEfficientFindNearestNeighbors(Instance instance, HashSet<Instance> kNeighbors, DistanceCalculator distCalc, Map<Instance,Double> map)
    {
        Instance secondInstance;
        InstanceComparator ic = new InstanceComparator();

        for (int i = 0; i < m_trainingInstances.size(); i++)
        {

            secondInstance = m_trainingInstances.instance(i);
            if(ic.compare(instance,secondInstance) != 0)
            {

                map.put(secondInstance, distCalc.distance(instance, secondInstance, p, 0));
            }
        }

        // get k neighbors
        for(int i = 0; i < k; i++)
        {
            kNeighbors.add(getAndRemoveMinKey(map, map.keySet()));
        }

        return kNeighbors;
    }


    /** getting the instance with min value, extracting him from map and returns him **/
    private Instance getAndRemoveMinKey(Map<Instance, Double> map, Set<Instance> keys) {
        Instance minKey = null;
        double value;
        double minValue = Double.MAX_VALUE;
        for(Instance key : keys)
        {
            value = map.get(key);
            if(value < minValue)
            {
                minValue = value;
                minKey = key;
            }
        }

        map.remove(minKey);
        return minKey;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     * @param
     * @return
     */
    public double getAverageValue (Instance instance, Iterator<Instance> it, double size)
    {
        double value = 0;
        while (it.hasNext())
        {
            value += it.next().value(m_trainingInstances.classIndex());
        }

        return (value/size);
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue(Instance instance, Iterator<Instance> it) {
        Instance inst;
        DistanceCalculator dcalc = new DistanceCalculator();
        dcalc.m_Efficient = distEffCheck;
        double numerator = 0;
        double denominator = 0;
        double wi;
        double dist = 0;

        while (it.hasNext())
        {
            inst = it.next();
            dist = dcalc.distance(instance, inst, p, m_kNeighborDist);
            if (dist == 0)
            {
                return inst.value(m_trainingInstances.classIndex());
            }

            wi = 1.0 / (Math.pow(dist,2));
            if (wi > 0)
            {
                numerator += wi * inst.value(m_trainingInstances.classIndex());
                denominator += wi;
            }
        }


        return (numerator/denominator);
    }

    public void setK(int k)
    {
        this.k = k;
    }

    public void setP(int p)
    {
        this.p = p;
    }

    public void setWeightingScheme(int weightingScheme)
    {
        m_weightingScheme = weightingScheme;
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