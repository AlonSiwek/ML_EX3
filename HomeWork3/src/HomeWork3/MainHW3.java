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
		Instances trainData = loadData("auto_price.txt");
		trainData.randomize(new Random()); // shuffle data ONCE !

		/** the diff in validation error of regular and efficient with the same fold
		 is tiny and negligible, the cause is probably the long/order of double calculations **/
//		tryAllCombinations(trainData);
		printFoldMessages(trainData);
	}


	private static void printFoldMessages(Instances trainData) throws Exception
	{
		//Instances[] arr = null;
		int[] numOfFolds = {trainData.size(), 50, 10, 5, 3};
		int num_of_folds;
		Knn knn;
		DistanceCalculator distanceCalculator;
		long[] avgFold = new long[2];

//		for(int i = 0; i < numOfFolds.length; i++)
//		{
//			num_of_folds = numOfFolds[i];
//			knn = new Knn(bestHyperParameteres[0], 2, false);
//			knn.buildClassifier(trainData);
//			double crossValidationError = knn.crossValidationError(trainData,num_of_folds);
//
//
//			System.out.println("--------------------------------" + "\n" +
//								"Results for " + num_of_folds + " folds:" +
//								"--------------------------------");
//			System.out.println("Cross validation error of regular knn on auto_price dataset is " + crossValidationError +
//								" and the average elapsed time is " + avgFold[0] + "\n" +
//								"The total elapsed time is: " + avgFold[1] + "\n");
//
//
//			System.out.println("Cross validation error of efficient knn on auto_price dataset is " + crossValidationError +
//								" and the average elapsed time is " + avgFold[0] + "\n" +
//								"The total elapsed time is: " + avgFold[1] + "\n");
//
//		}

		distanceCalculator = new DistanceCalculator(1, true);
		knn = new Knn(3,false, distanceCalculator);
		System.out.println(knn.crossValidationError(trainData, 10));



	}

}