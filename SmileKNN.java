
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.classification.KNN;

public class SmileKNN {

	public static void main(String[] args) {
		try {
			ArffParser p = new ArffParser();
			p.setResponseIndex(4);
			AttributeDataset data = p.parse(args[0]);
			double[][] x = data.x();
			int[] labels = data.labels();
			KNN knn = KNN.learn(x, labels, 3);
			int[] pred = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				pred[i] = knn.predict(x[i]);
			}
			double trainError = error(pred, labels);

			System.out.format("training error = %.2f%%\n", 100*trainError);
		}
		catch (Exception ex) { ex.printStackTrace(); }
	}
	
	/**
     * Returns the error rate.
     */
    static double error(int[] x, int[] y) {
        int e = 0;

        for (int i = 0; i < x.length; i++) {
            if (x[i] != y[i]) {
                e++;
            }
        }

        return (double) e / x.length;
    }
}