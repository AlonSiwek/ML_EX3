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
		int[] originalDataBestParams = new int[3];
		int[] scaledDataBestParams = new int[3];
		double minOriginalDataErr = Double.MAX_VALUE;
		double minScaledDataErr = Double.MAX_VALUE;
		double currOriginalDataErr = Double.MAX_VALUE;
		double currScaledDataErr = Double.MAX_VALUE;
		FeatureScaler featureScaler = new FeatureScaler();
		Instances originalData = loadData("auto_price.txt");
		Instances scaledData = featureScaler.scaleData(originalData);
		String majorityFunction;

		// shuffle data for cross validation
		originalData.randomize(new Random());
		scaledData.randomize(new Random());

//		// iterate over num of neighbours
		for (int k = 1; k <= 20; k++) {

			// iterate over different p's for lp distance calcs
			for (int p : pArray) {

				// toggle weighting method
				for (int j = 0; j < 2; j++) {
					boolean weighting = (j % 2 == 0);
					distanceCalculator = new DistanceCalculator(p, false);
					knn = new Knn(k, weighting, distanceCalculator);

					// calculate cross validation avg error on original data and update hyper params if needed
					currOriginalDataErr = knn.crossValidationError(originalData, 10);

					if (currOriginalDataErr < minOriginalDataErr) {
						minOriginalDataErr = currOriginalDataErr;
						originalDataBestParams[0] = k;
						originalDataBestParams[1] = p;
						originalDataBestParams[2] = j;
					}

					// calculate cross validation avg error on scaled data and update hyper params if needed
					currScaledDataErr = knn.crossValidationError(scaledData, 10);
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


		int[] numOfFolds = {originalData.size(), 50, 10, 5, 3};
		// TODO iterate over nunOfFolds <yellow part of ex>




	}


}