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
		Instances trainData = loadData("auto_price.txt");

		// shuffle data for cross validation
		trainData.randomize(new Random());

		int[] pArray = {1,2,3,Integer.MAX_VALUE};

		for (int k = 1; k <= 20; k++) {
			for (int p : pArray) {
				for (int j = 0; j < 2; j++) {
					boolean weighting = (j % 2 == 0);
					distanceCalculator = new DistanceCalculator(p, false);

					knn = new Knn(k, weighting, distanceCalculator);
//					System.out.println(knn.crossValidationError(trainData, 10));
				}
			}
		}
		
		int[] numOfFolds = {trainData.size(), 50, 10, 5, 3};

		distanceCalculator = new DistanceCalculator(1, false);

		knn = new Knn(1, true, distanceCalculator);
		System.out.println(knn.crossValidationError(trainData, 10));


	}


}