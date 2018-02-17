import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.clustering.BIRCH;

public class SmileBIRCH {

	public static void main(String[] args) {
		try {
			ArffParser p = new ArffParser();
			p.setResponseIndex(2);
			AttributeDataset data = p.parse(args[0]);
			double[][] x = data.x();
			int[] labels = data.labels();
			int clusterNum = 6;
			
			BIRCH birch = new BIRCH(2,5,0.5);
			for (int i=0; i<x.length; i++) {
				birch.add(x[i]);
			}
			birch.partition(clusterNum,5);
			int[] pred = new int[labels.length];
			for (int i = 0; i < labels.length; i++) {
				pred[i] = birch.predict(x[i]);
			}
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