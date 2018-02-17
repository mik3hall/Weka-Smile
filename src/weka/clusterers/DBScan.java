package weka.clusterers;

import java.beans.PropertyEditor;
import java.beans.PropertyEditorManager;
import java.util.List;
import java.util.Vector;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Properties;

import java.lang.reflect.Array;
import weka.Run;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
//import weka.core.DistanceFunction;
import weka.core.Option;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.gui.GenericPropertiesCreator;

import smile.clustering.Clustering;
import us.hall.weka.smile.SmileDistance;
import us.hall.weka.smile.SmileEuclideanDistance;

/**
 * <!-- globalinfo-start --> Cluster data using the Smile DBScan algorithm. Can use
 * either the Euclidean distance (default) or the Manhattan distance. 
 * For more information see:<br/>
 * @see <a href="http://haifengl.github.io/smile/">Smile DBScan</a>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -M &lt;num&gt;
 *  The minimum number of neighbors for a core data point.
 *  (default 2).
 * </pre>
 * 
 * <pre>
 * -A &lt;classname and options&gt;
 *  Distance function to use.
 *  (default: us.hall.weka.smile.SmileEuclideanDistance)
 * </pre>
 * 
 * <pre>
 * -R
 *  The neighborhood radius.
 * </pre>
 * 
 * <pre>
 * -do-not-check-capabilities
 *  If set, clusterer capabilities are not checked before clusterer is built
 *  (use with caution).
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Mike Hall (mik3hall@gmail.com)
 * @see RandomizableClusterer
 */
 
public class DBScan extends RandomizableClusterer {

	static {
//		PropertyEditorManager.registerEditor(smile.math.distance.Distance.class,
		PropertyEditorManager.registerEditor(us.hall.weka.smile.SmileDistance.class,
			weka.gui.GenericObjectEditor.class);
	}
		
	/**
	 * Smile DBScan
	 */
	transient smile.clustering.DBScan m_dbscan;
	
	/**
	 * The minimum number of neighbors for a core data point.
	 */
	int m_min = 2;
	
	/**
	 * The neighborhood radius.
	 */
	double m_range = 5;
	
	/**
	 * Distance function
	 */
	SmileDistance m_dist = new SmileEuclideanDistance();	// .getSmileDistance();
	
  	/** the distance function used. */
//  	protected DistanceFunction m_DistanceFunction = new weka.core.EuclideanDistance();
  	
  /**
   * number of clusters to generate.
   */
  protected int m_NumClusters = 2;
  	
  /**
   * replace missing values in training instances.
   */
  protected ReplaceMissingValues m_ReplaceMissingFilter;

  /**
   * Replace missing values globally?
   */
  protected boolean m_dontReplaceMissing = false;

	/**
	 * Default constructor
	 */
	 public DBScan() {
//	 	checkGOEProps();		// For Distance algorithm list choose button
	 }
	 
	  /**
	   * Returns default capabilities of the clusterer.
	   * 
	   * @return the capabilities of this clusterer
	   */
	  @Override
	  public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NO_CLASS);

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		return result;
	  }
  
	public void buildClusterer(Instances data) throws Exception {
		// can clusterer handle the data?
		getCapabilities().testWithFail(data);

		m_ReplaceMissingFilter = new ReplaceMissingValues();
		Instances instances = new Instances(data);

		instances.setClassIndex(-1);		
		
		if (!m_dontReplaceMissing) {
		  m_ReplaceMissingFilter.setInputFormat(instances);
		  instances = Filter.useFilter(instances, m_ReplaceMissingFilter);
		}
		Object[] oA = asArrays(data,instances.numAttributes()-1);
		double[][] idata = (double[][])oA[0];
		m_dbscan = new smile.clustering.DBScan<>(idata,m_dist.getSmileDistance(),m_min,m_range);
		m_NumClusters = m_dbscan.getNumClusters();
		m_NumClusters++;			// Allow for outliers to be considered a cluster
	}

	public int clusterInstance(Instance instance) throws Exception {
		try {
			Instance inst = null;
			if (!m_dontReplaceMissing) {
			  m_ReplaceMissingFilter.input(instance);
			  m_ReplaceMissingFilter.batchFinished();
			  inst = m_ReplaceMissingFilter.output();
			} else {
			  inst = instance;
			}
			double[] dA = new double[instance.numAttributes()-1];
			for (int i = 0; i < instance.numAttributes()-1; i++) {
				if (i == instance.classIndex()) continue;
				dA[i] = instance.value(i);
			}
			int p = m_dbscan.predict(dA);
			if (p == Clustering.OUTLIER) {
				return m_NumClusters-1;
			}
			return p;
		}
		catch (Throwable tossed) { tossed.printStackTrace(); }
		return 0;
	}

  public Object[] asArrays(Instances data, int classIndex) {
  	double dA[][] = new double[data.numInstances()][data.numAttributes()-1 ];
  	int cA[] = new int[data.numInstances()];
	for (int i = 0; i < data.numInstances(); i++) {
		for (int j = 0; j < data.numAttributes(); j++) {
			if (j == classIndex) {
				cA[i] = (int)data.instance(i).value(j);
			}
			else dA[i][j] = data.instance(i).value(j);
		}
	}
	return new Object[] {dA, cA};
  } 
  	
  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> result = new Vector<Option>();

    result.addElement(new Option("\tThe minimum number of neighbors for a core data point.\n" + "\t(default 2).",
      "N", 1, "-N <num>"));

	result.addElement(new Option("\tDistance function to use.\n"
      + "\t(default: us.hall.weka.smile.SmileDistance)", "A", 1,
      "-A <classname and options>"));
      
    result.addElement(new Option("\tThe neighborhood radius.\n" + "\t(default 5).",
      "N", 1, "-N <num>"));
   
    result.addAll(Collections.list(super.listOptions()));

    return result.elements();
  }
   
  /**
   * Returns the number of clusters.
   * 
   * @return the number of clusters generated for a training dataset.
   * @throws Exception if number of clusters could not be returned successfully
   */
   @Override
   public int numberOfClusters() throws Exception {
   	return m_NumClusters;
   }
	
   /**
    * Parses a given list of options.
    * <p/>
    * 
    * <!-- options-start --> Valid options are:
    * <p/>
    * 
    * <pre>
    * -M &lt;num&gt;
    *  The minimum number of neighbors for a core data point.
    *  (default 2).
    * </pre>
    * 
    * <pre>
    * -A &lt;classname and options&gt;
    *  Distance function to use.
    *  (default: us.hall.weka.smile.SmileEuclideanDistance)
    * </pre>
    * 
    * <pre>
    * -R &lt;num&gt;
    *  The neighborhood radius.
    * </pre>
    * 
    * <!-- options-end -->
    * 
    * @param options the list of options as an array of strings
    * @throws Exception if an option is not supported
    */
    @Override
    public void setOptions(String[] options) throws Exception {

		String temp = Utils.getOption("M", options);
		if (temp.length() > 0) {
		  setMinimum(Integer.parseInt(temp));
		}
    
    	temp = Utils.getOption("R",options);
    	if (temp.length() > 0) {
    		setRange(Double.parseDouble(temp));
    	}
    	
    	String distFunctionClass = Utils.getOption('A', options);
		if (distFunctionClass.length() != 0) {
			String distFunctionClassSpec[] = Utils.splitOptions(distFunctionClass);
/*
			for (String spec : distFunctionClassSpec) {
				System.out.println("spec="+spec);
			}
			if (distFunctionClassSpec.length == 0) {
				throw new Exception("Invalid Distance specification string.");
			}
*/
			String className = distFunctionClassSpec[0];
			distFunctionClassSpec[0] = "";
			setDistance((SmileDistance) Utils.forName(
				SmileDistance.class, className, distFunctionClassSpec));
		} 
    	super.setOptions(options);

    	Utils.checkForRemainingOptions(options);    	
    }
    
/*
    private static void checkGOEProps() {
    	System.out.println("cjeckGOEProps");
    	// If needed add a GenericObjectEditor editor for Smile distance classes property
		// This will give you a choose button for the different distance functions
		// Currently it seems an incomplete list and some don't work
		// NOTE: Not all of them seem compatible with ClassDiscovery. Using a hardcoded
		//       list of what seems like should generally work
		Properties GPCInputProps =
			GenericPropertiesCreator.getGlobalInputProperties();
		String key = "smile.math.distance.Distance";
		StringBuilder value = new StringBuilder("us.hall.weka.smile.SmileEuclideanDistance,");
		value.append("smile.math.distance.ChebyshevDistance,");
		value.append("smile.math.distance.ManhattanDistance,");
		value.append("smile.math.distance.MinkowskiDistance");
		if (GPCInputProps != null) {
			GPCInputProps.put(key, value.toString());
		}
		else System.out.println("GPCInputProps null");
    }
*/

  /**
   * Gets the current settings of DBScan.
   * 
   * @return an array of strings suitable for passing to setOptions()
   */
  @Override
  public String[] getOptions() {
    Vector<String> result = new Vector<String>();

    result.add("-M");
    result.add("" + getMinimum());

	result.add("-R");
	result.add("" + getRange());
	
    result.add("-A");
    result.add(m_dist.getClass().getName());	// + " " + Utils
//      .joinOptions(m_dist.getOptions())).trim());

    Collections.addAll(result, super.getOptions());

    return result.toArray(new String[result.size()]);
  }
    /**
    
     * Set the minimum number of neighbors for a core data point.
     * @param the minimum
     */
    public void setMinimum(int min) {
    	m_min = min;
    }
    
    public int getMinimum() {
    	return m_min;
    }
    
    /**
     * Set the neighborhood radius
     * @param the radius
     */
    public void setRange(double range) {
    	m_range = range;
    }
    
    public double getRange() {
    	return m_range;
    }
    
    /**
     * Set the distance function to use
     * @param the distance function
     */
    public void setDistance(SmileDistance dist) {
    	m_dist = dist;
    }
    
    public SmileDistance getDistance() {
    	return m_dist;
    }
    
  /**
   * Main method for executing this class.
   * 
   * @param args use -h to list all parameters
   */
  public static void main(String[] args) {
    runClusterer(new DBScan(), args);
  }
}