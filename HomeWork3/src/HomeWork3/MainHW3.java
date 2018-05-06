package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Knn knn;
		DistanceCalculator distanceCalculator;
		int[] pArray = {1,2,3,Integer.MAX_VALUE};
		int[] originalDataBestParams = new int[3]; //{k,p,weighing}
		int[] scaledDataBestParams = new int[3]; // {k,p,weighing}
		double minOriginalDataErr = Double.MAX_VALUE;
		double minScaledDataErr = Double.MAX_VALUE;
		double currOriginalDataErr;
		double currScaledDataErr;
		FeatureScaler featureScaler = new FeatureScaler();
		Instances originalData = loadData("auto_price.txt");
		Instances scaledData = featureScaler.scaleData(originalData);
		Instances[][] separatedData;
		String majorityFunction;

		// shuffle data for cross validation
		originalData.randomize(new Random());
		scaledData.randomize(new Random());

		// iterate over num of neighbours
		for (int k = 1; k <= 20; k++) {

			// iterate over different p's for lp distance calcs
			for (int p : pArray) {

				// toggle weighting method
				for (int j = 0; j < 2; j++) {
					boolean weighting = (j == 0);
					distanceCalculator = new DistanceCalculator(p, false);
					knn = new Knn(k, weighting, distanceCalculator);

					// calculate cross validation avg error on original data and update hyper params if needed
					separatedData = knn.separateData(originalData, 10);
					currOriginalDataErr = knn.crossValidationError(separatedData, 10);

					if (currOriginalDataErr < minOriginalDataErr) {
						minOriginalDataErr = currOriginalDataErr;
						originalDataBestParams[0] = k;
						originalDataBestParams[1] = p;
						originalDataBestParams[2] = j;
					}

					// calculate cross validation avg error on scaled data and update hyper params if needed
					separatedData = knn.separateData(scaledData, 10);
					currScaledDataErr = knn.crossValidationError(separatedData, 10);
					if (currScaledDataErr < minScaledDataErr) {
						minScaledDataErr = currScaledDataErr;
						scaledDataBestParams[0] = k;
						scaledDataBestParams[1] = p;
						scaledDataBestParams[2] = j;
					}
				}
			}
		}

		// print original data best params and error
		majorityFunction = (originalDataBestParams[2] == 0) ? "weighted" : "regular" ;
		System.out.println("----------------------------\n" +
				"Results for original dataset:\n" +
				"----------------------------\n" +
				"Cross validation error with K = " + originalDataBestParams[0] + ", " +
				"lp = " + originalDataBestParams[1] +", " +
				"majority function = " + majorityFunction +
				" for auto_price data is: " + minOriginalDataErr);
		System.out.println();

		// print scaled data best params and error
		majorityFunction = (scaledDataBestParams[2] == 0) ? "weighted" : "regular" ;
		System.out.println("----------------------------\n" +
				"Results for original dataset:\n" +
				"----------------------------\n" +
				"Cross validation error with K = " + scaledDataBestParams[0] + ", " +
				"lp = " + scaledDataBestParams[1] +", " +
				"majority function = " + majorityFunction +
				" for auto_price data is: " + minScaledDataErr);
		System.out.println();

		//time measurements
		int[] numOfFolds = {scaledData.size(), 50, 10, 5, 3};
		boolean weighting = (scaledDataBestParams[2] == 0);
		long startTime;
		long timeElapsed;
		double error;

		DistanceCalculator efficientDistanceCalculator = new DistanceCalculator(scaledDataBestParams[1], true);
		DistanceCalculator regulartDistanceCalculator = new DistanceCalculator(scaledDataBestParams[1], false);

		Knn efficientKnn = new Knn(scaledDataBestParams[0], weighting, efficientDistanceCalculator);
		Knn regularKnn = new Knn(scaledDataBestParams[0], weighting, regulartDistanceCalculator);

		for (int folds : numOfFolds) {

			System.out.println("----------------------------");
			System.out.println("Results for " + folds + " folds:");
			System.out.println("----------------------------");

			// separate data
			separatedData = regularKnn.separateData(scaledData, folds);

			// measure time for regular prediction
		 	startTime = System.nanoTime();
			error = regularKnn.crossValidationError(separatedData, folds);
			timeElapsed = System.nanoTime() - startTime;

			// print results
			System.out.println("Cross validation error of regular knn on auto_price dataset is:" + error);
			System.out.println("the average elapsed time is " + (timeElapsed / ((float) folds)));
			System.out.println("The total elapsed time is: " + timeElapsed);
			System.out.println();

			// separate data
			separatedData = efficientKnn.separateData(scaledData, folds);

			// measure time for efficient prediction
			startTime = System.nanoTime();
			error = efficientKnn.crossValidationError(separatedData, folds);
			timeElapsed = System.nanoTime() - startTime;

			// print results
			System.out.println("Cross validation error of efficient knn on auto_price dataset is:" + error);
			System.out.println("the average elapsed time is " + (timeElapsed / ((float) folds)));
			System.out.println("The total elapsed time is: " + timeElapsed);
			System.out.println();
		}
	}
}