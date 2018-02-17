package us.hall.weka.smile;

import java.io.Serializable;

import smile.math.distance.Distance;
import smile.math.distance.EuclideanDistance;
import weka.core.Instances;


/**
 * Weka wrapper class for Smile EuclideanDistance class
 * that will work with Weka ClassDiscovery
 */
public class SmileEuclideanDistance implements SmileDistance, Serializable {
	
	Instances data;
	
  /**
   * Constructs an Euclidean Distance object, Instances must be still set.
   */
  public SmileEuclideanDistance() {
  }

  /**
   * Constructs an Euclidean Distance object and automatically initializes the
   * ranges.
   * 
   * @param data 	the instances the distance function should work on
   */
  public SmileEuclideanDistance(Instances data) {
    this.data = data;
  }	
  
  public Distance getSmileDistance() {
  	return new EuclideanDistance();
  }
}