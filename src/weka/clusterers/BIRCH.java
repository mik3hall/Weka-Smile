package weka.clusterers;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.WekaPackageClassLoaderManager;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import smile.clustering.Clustering;

/**
 * <!-- globalinfo-start --> Cluster data using the Smile BIRCH algorithm.
 * For more information see:<br/>
 * @see <a href="http://haifengl.github.io/smile/">Smile BIRCH</a>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -k &lt;num&gt;
 *  The cluster number.
 * </pre>
 * 
 * <pre>
 * -B &lt;num&gt;
 *  The branching factor. Maximum number of children nodes.
 * </pre>
 * 
 * <pre>
 * -R &lt;num&gt; 
 *  The maximum radius of a sub-cluster.
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
 
public class BIRCH extends RandomizableClusterer {

	/**
	 * Smile BIRCH
	 */
	transient smile.clustering.BIRCH m_birch;
	
	/**
	 * The branching factor. Maximum number of children nodes.
	 */
	int m_branch = 5;

	/**
	 * The maximum radius of a sub-cluster.
	 */
	double m_radius = 16.0;
	
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
		m_birch = new smile.clustering.BIRCH(idata[0].length,m_branch,m_radius);
		for (int i = 0; i < idata.length; i++)
            m_birch.add(idata[i]);   
        m_birch.partition(numberOfClusters(),5);        
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
			int p = m_birch.predict(dA);
			if (p == Clustering.OUTLIER) {
				return 0;
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
   * Returns the number of clusters.
   * 
   * @return the number of clusters generated for a training dataset.
   */
   @Override
   public int numberOfClusters() {
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
    * -k &lt;num&gt;
    *  The dimensionality of data. (Cluster number).
    * </pre>
    * 
    * <pre>
    * -B &lt;num&gt;
    *  The branching factor. Maximum number of children nodes.
    * </pre>
    * 
    * <pre>
    * -R &lt;num&gt; 
    *  The maximum radius of a sub-cluster.
    * </pre>
    * 
    * <!-- options-end -->
    * 
    * @param options the list of options as an array of strings
    * @throws Exception if an option is not supported
    */
    @Override
    public void setOptions(String[] options) throws Exception {
		String temp = Utils.getOption("k", options);
		if (temp.length() > 0) {
		  setNumClusters(Integer.parseInt(temp));
		}
    
    	temp = Utils.getOption("B",options);
    	if (temp.length() > 0) {
    		setBranch(Integer.parseInt(temp));
    	}
    	
    	temp = Utils.getOption("T",options);
    	if (temp.length() > 0) {
    		setRadius(Double.parseDouble(temp));
    	}
 		
    	super.setOptions(options);

    	Utils.checkForRemainingOptions(options);    	
    }
    
  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> result = new Vector<Option>();
    result.addElement(new Option("\tThe cluster number.\n" + "\t(default 2).",
      "N", 1, "-N <num>"));
      
    result.addElement(new Option("\tThe branching factor. Maximum number of children nodes.\n",
      "N", 1, "-N <num>"));
      
    result.addElement(new Option("\tThe neighborhood radius.\n" + "\t(default 5).",
      "N", 1, "-N <num>"));
   
    result.addAll(Collections.list(super.listOptions()));

    return result.elements();
  }
  
  /**
   * Gets the current settings of BIRCH.
   * 
   * @return an array of strings suitable for passing to setOptions()
   */
  @Override
  public String[] getOptions() {
    Vector<String> result = new Vector<String>();

	result.add("-k");
	result.add("" + numberOfClusters());
	
    result.add("-B");
    result.add("" + getBranch());

	result.add("-T");
	result.add("" + getRadius());

    Collections.addAll(result, super.getOptions());
    return result.toArray(new String[result.size()]);
  }
      
    /**
     * Set the minimum number of neighbors for a core data point.
     * @param the minimum
     */
    public void setNumClusters(int k) {
    	m_NumClusters = k;
    }
    
    public int getNumClusters() {
    	return m_NumClusters;
    }
    
    /**
     * Set the minimum number of neighbors for a core data point.
     * @param the minimum
     */
    public void setBranch(int branch) {
    	m_branch = branch;
    }
    
    public int getBranch() {
    	return m_branch;
    }
    
    /**
     * Set the distance function to use
     * @param the distance function
     */
    public void setRadius(double radius) {
    	m_radius = radius;
    }
    
    public double getRadius() {
    	return m_radius;
    }
    
  /**
   * Main method for executing this class.
   * 
   * @param args use -h to list all parameters
   */
  public static void main(String[] args) {
    runClusterer(new BIRCH(), args);
  }
}